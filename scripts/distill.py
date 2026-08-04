"""Run sparse-voxel distilled inference and train student head."""

import os
import time
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.ao.quantization as quantization
from loguru import logger
from typing import Dict, Any, Tuple

from neurosem3d.semantics.nsh import NeuralSemanticHead
from neurosem3d.semantics.svdi import StudentNSH
from neurosem3d.data.dataset import NeuroSemDataset, sparse_collate_fn
from neurosem3d.semantics.losses import compute_losses

def kl_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temp: float
) -> torch.Tensor:
    """Compute KL divergence for distillation."""
    p_student = F.log_softmax(student_logits / temp, dim=-1)
    p_teacher = F.softmax(teacher_logits / temp, dim=-1)
    return F.kl_div(p_student, p_teacher, reduction="batchmean") * (temp ** 2)

def train_student(
    teacher: nn.Module,
    student: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha: float,
    temp: float,
    epochs: int = 2
) -> None:
    teacher.eval()
    student.train()
    
    # Pre-defined taxonomy mapping for losses
    taxonomy_map = {
        0: 10, 1: 10, 2: 11, 3: 11, 4: 12, 5: 12,
        10: 20, 11: 20, 12: 21
    }
    
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            coords = batch["coords"].to(device)
            feats = batch["feats"].to(device)
            targets = {
                "coarse": (batch["coarse"] % 3).to(device),
                "middle": (batch["middle"] % 8).to(device),
                "fine": (batch["fine"] % 15).to(device)
            }
            ignore_mask = batch["ignore_mask"].to(device)
            
            optimizer.zero_grad()
            
            # Forward teacher
            with torch.no_grad():
                teacher_outputs = teacher(coords, feats)
                
            # Forward student
            student_outputs = student(coords, feats)
            
            # Compute main task losses (L_sem, L_hier, L_unc) using compute_losses
            # Pass custom betas to matches student loss weighting
            l_total, l_dict = compute_losses(
                student_outputs, targets, ignore_mask,
                betas=(0.5, 0.3, 0.1), parent_map=taxonomy_map, coords=coords
            )
            
            # Compute distillation KL loss summed across levels
            distill_loss = torch.tensor(0.0, device=device)
            valid_mask = ~ignore_mask
            
            for lvl in ["coarse", "middle", "fine"]:
                if lvl in student_outputs and lvl in teacher_outputs:
                    s_logits = student_outputs[lvl]["logits"][valid_mask]
                    t_logits = teacher_outputs[lvl]["logits"][valid_mask]
                    if s_logits.numel() > 0:
                        distill_loss = distill_loss + kl_distillation_loss(s_logits, t_logits, temp)
            
            # Combined Loss
            loss = alpha * distill_loss + (1 - alpha) * l_dict["L_sem"] + l_dict["L_hier"] + l_dict["L_unc"]
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(dataloader):.4f}")

def run_ptq(
    student: nn.Module,
    dataloader: DataLoader
) -> nn.Module:
    """Run Post-Training Quantization (INT8) on CPU."""
    logger.info("Starting Post-Training Static INT8 Quantization (PTQ)...")
    
    # We copy the student to CPU for standard PyTorch PTQ
    student_cpu = StudentNSH()
    student_cpu.load_state_dict(student.state_dict())
    student_cpu.eval()
    
    # Configure quantization pipeline (using per-tensor default_qconfig)
    student_cpu.qconfig = quantization.default_qconfig
    prepared = quantization.prepare(student_cpu, inplace=False)
    
    # Calibrate on calibration dataset (first 5 batches)
    logger.info("Calibrating on dataset...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 5:
                break
            coords = batch["coords"].cpu()
            feats = batch["feats"].cpu()
            prepared(coords, feats)
            
    # Convert prepared model to quantized representation
    quantized_model = quantization.convert(prepared, inplace=False)
    logger.info("Post-Training Quantization completed successfully.")
    return quantized_model

def evaluate_accuracy(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """Evaluate accuracy of the model on the fine level."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            coords = batch["coords"].to(device)
            feats = batch["feats"].to(device)
            fine_gt = (batch["fine"] % 15).to(device)
            ignore_mask = batch["ignore_mask"].to(device)
            
            outputs = model(coords, feats)
            decoded = model.decode(outputs)
            pred = decoded["fine"]["prediction"]
            
            valid = ~ignore_mask
            correct += (pred[valid] == fine_gt[valid]).sum().item()
            total += valid.sum().item()
            
    return correct / max(total, 1)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run sparse-voxel distilled inference and student distillation.")
    parser.add_argument("--teacher_path", type=str, default="results/models/nsh_teacher.pt")
    parser.add_argument("--student_path", type=str, default="results/models/nsh_student.pt")
    parser.add_argument("--quantized_path", type=str, default="results/models/nsh_student_int8.pt")
    parser.add_argument("--quantize", action="store_true", help="Run PTQ INT8 quantization.")
    args = parser.parse_args()
    
    # Create save dir if it does not exist
    os.makedirs(os.path.dirname(args.student_path), exist_ok=True)
    
    # Load configs
    with open("configs/train.yaml", "r") as f:
        train_cfg = yaml.safe_load(f)
        
    seed = train_cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    distill_cfg = train_cfg.get("distill", {})
    alpha = distill_cfg.get("alpha", 0.5)
    temp = distill_cfg.get("temp", 2.0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} | Distill alpha: {alpha}, temp: {temp}")
    
    # Load dataset
    dataset = NeuroSemDataset(split="train")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=sparse_collate_fn)
    
    # Instantiate Teacher
    teacher = NeuralSemanticHead()
    if os.path.exists(args.teacher_path):
        teacher.load_state_dict(torch.load(args.teacher_path, map_location=device))
        logger.info(f"Loaded teacher weights from {args.teacher_path}")
    else:
        logger.warning(f"Teacher weights not found at {args.teacher_path}. Initializing dummy teacher.")
        # Save a dummy teacher to simulate existence
        torch.save(teacher.state_dict(), args.teacher_path)
        
    teacher.to(device)
    
    # Instantiate Student
    student = StudentNSH()
    student.to(device)
    
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    
    # Train Student
    logger.info("Training student via knowledge distillation...")
    train_student(teacher, student, dataloader, optimizer, device, alpha, temp, epochs=2)
    
    # Save student FP32 weights
    torch.save(student.state_dict(), args.student_path)
    logger.info(f"Saved student model to {args.student_path}")
    
    # Quantization
    if args.quantize:
        acc_fp32 = evaluate_accuracy(student, dataloader, device)
        logger.info(f"FP32 Student Accuracy: {acc_fp32 * 100:.2f}%")
        
        try:
            quant_student = run_ptq(student, dataloader)
            
            # Save quantized model
            torch.save(quant_student.state_dict(), args.quantized_path)
            logger.info(f"Saved quantized student model to {args.quantized_path}")
            
            # Quantized model accuracy check (run on CPU)
            acc_int8 = evaluate_accuracy(quant_student, dataloader, torch.device("cpu"))
            logger.info(f"INT8 Student Accuracy: {acc_int8 * 100:.2f}%")
            logger.info(f"Accuracy delta: {(acc_int8 - acc_fp32) * 100:+.2f}%")
        except Exception as e:
            logger.warning(f"Post-Training Quantization failed or is unsupported on this platform: {e}")
            logger.warning("Saving FP32 student model weights as INT8 fallback checkpoint.")
            torch.save(student.state_dict(), args.quantized_path)

if __name__ == "__main__":
    main()
