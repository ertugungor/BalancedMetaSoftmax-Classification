import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import json
from loss import BalancedSoftmaxLoss
from utils import source_import

BALANCED_SOFTMAX_DEF_FILE = "/home/user502/dev/dive/BalancedMetaSoftmax-Classification/loss/BalancedSoftmaxLoss.py"

class DiveLoss(_Loss):
  def __init__(self, freq_path, weight, temperature, power_norm):
    super(DiveLoss, self).__init__()
    self.weight = weight
    self.temperature = temperature
    self.power_norm = power_norm
    self.balanced_softmax_loss = source_import(BALANCED_SOFTMAX_DEF_FILE).create_loss(freq_path).cuda()
    # self.balanced_softmax_loss = BalancedSoftmaxLoss(freq_path)

  def forward(self, student_logits, teacher_logits, labels):
    balanced_softmax = (1-self.weight) * self.balanced_softmax_loss.forward(student_logits, labels)
    transformed_teacher_logits = self.transform_teacher_logits(teacher_logits)
    transformed_student_logits = self.transform_student_logits(student_logits)
    teacher_student_kl_div = self.weight * (self.temperature ** 2) * self.kl_div(transformed_teacher_logits, transformed_student_logits)
    return balanced_softmax + teacher_student_kl_div

  def transform_teacher_logits(self, teacher_logits):
    temp_teacher_logits = F.softmax(teacher_logits / self.temperature)
    pow_teacher_logits = torch.pow(temp_teacher_logits, self.power_norm)
    return pow_teacher_logits / torch.sum(pow_teacher_logits)

  def transform_student_logits(self, student_logits):
    return F.softmax(student_logits / self.temperature)

  def kl_div(self, transformed_teacher_logits, transformed_student_logits):
    return F.kl_div(transformed_teacher_logits, transformed_student_logits, reduction="batchmean")

def create_loss(freq_path, weight, temperature, power_norm):
  print('Loading Dive Loss..')
  return DiveLoss(freq_path, weight, temperature, power_norm)
