import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    """
    Binary Cross-Entropy Loss for prediction-level distillation.
    """
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, y_teacher, y_student):
        """
        Args:
            y_teacher (torch.Tensor): Soft labels from the Teacher model (pseudo-labels).
            y_student (torch.Tensor): Predictions from the Student model.
        Returns:
            torch.Tensor: BCE Loss value.
        """
        return self.bce(y_student, y_teacher)


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss for feature-level distillation.
    """
    def __init__(self, temperature=0.1):
        """
        Args:
            temperature (float): Temperature parameter for scaling similarity scores.
        """
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_teacher_anomaly, z_student_anomaly, z_teacher_normal, z_student_normal):
        """
        Args:
            z_teacher_anomaly (torch.Tensor): Anomalous features from Teacher.
            z_student_anomaly (torch.Tensor): Anomalous features from Student.
            z_teacher_normal (torch.Tensor): Normal features from Teacher.
            z_student_normal (torch.Tensor): Normal features from Student.
        Returns:
            torch.Tensor: InfoNCE Loss value.
        """
        # Cosine similarity
        sim = lambda x, y: F.cosine_similarity(x, y, dim=-1)

        # Anomalous term
        pos_sim_anomaly = sim(z_teacher_anomaly, z_student_anomaly) / self.temperature
        neg_sim_anomaly = sim(z_teacher_anomaly, z_student_normal) / self.temperature
        neg_exp_anomaly = torch.exp(neg_sim_anomaly).sum(dim=0)
        loss_anomaly = -torch.log(torch.exp(pos_sim_anomaly) / (neg_exp_anomaly + torch.exp(pos_sim_anomaly)))

        # Normal term
        pos_sim_normal = sim(z_teacher_normal, z_student_normal) / self.temperature
        neg_sim_normal = sim(z_teacher_normal, z_student_anomaly) / self.temperature
        neg_exp_normal = torch.exp(neg_sim_normal).sum(dim=0)
        loss_normal = -torch.log(torch.exp(pos_sim_normal) / (neg_exp_normal + torch.exp(pos_sim_normal)))
 
        # Total InfoNCE loss
        return (loss_anomaly.mean() + loss_normal.mean()) / 2 # chia 2 để giảm bớt độ chi phối 2 loss đồng thời giảm  Learning rate. Không bắt buộc chia 2
        #return loss_anomaly.mean() + loss_normal.mean() #code not divide 2

class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation.
    """
    def __init__(self, alpha=0.5, temperature=0.1):
        """
        Args:
            alpha (float): Scaling factor for balancing BCE and InfoNCE losses.
            temperature (float): Temperature parameter for InfoNCE Loss.
        """
        super(DistillationLoss, self).__init__()
        self.bce_loss = BCELoss()
        self.nce_loss = InfoNCELoss(temperature)
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, y_teacher, y_student, z_teacher_anomaly, z_student_anomaly, z_teacher_normal, z_student_normal):
        """
        Args:
            y_teacher (torch.Tensor): Soft labels from Teacher (pseudo-labels).
            y_student (torch.Tensor): Predictions from Student.
            z_teacher_anomaly (torch.Tensor): Anomalous features from Teacher.
            z_student_anomaly (torch.Tensor): Anomalous features from Student.
            z_teacher_normal (torch.Tensor): Normal features from Teacher.
            z_student_normal (torch.Tensor): Normal features from Student.
        Returns:
            torch.Tensor: Combined distillation loss value.
        """
        bce_loss = self.bce_loss(y_teacher, y_student)
        nce_loss = self.nce_loss(z_teacher_anomaly, z_student_anomaly, z_teacher_normal, z_student_normal)
        return bce_loss + self.alpha * (self.temperature ** 2) * nce_loss

# Example usage
if __name__ == "__main__":
    # create input:
    y_teacher = torch.rand(32, 1)  # Soft labels from Teacher (pseudo-labels)
    y_student = torch.rand(32, 1)  # Predictions from Student
    z_teacher_anomaly = torch.rand(16, 128)  # Anomalous features from  Teacher
    z_student_anomaly = torch.rand(16, 128)  # Anomalous features from  Student
    z_teacher_normal = torch.rand(16, 128)  # Normal features from Teacher
    z_student_normal = torch.rand(16, 128)  # Normal features from Student

    # create loss function
    distillation_loss = DistillationLoss(alpha=0.5, temperature=0.1)

    # calutation loss.
    loss = distillation_loss(y_teacher, y_student, z_teacher_anomaly, z_student_anomaly, z_teacher_normal, z_student_normal)
    print(f"Distillation Loss: {loss.item()}")
