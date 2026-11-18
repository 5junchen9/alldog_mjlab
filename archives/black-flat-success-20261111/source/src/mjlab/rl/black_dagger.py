"""DAgger动作数据集：均匀保留历史教师标注，避免内存无限增长。"""

import torch


class DAggerDataset:
  def __init__(self, capacity: int = 100000) -> None:
    self.capacity = capacity
    self.total_seen = 0
    self.history = torch.empty(0, 5, 45)
    self.teacher_action = torch.empty(0, 12)
    self.priorities = torch.empty(0, dtype=torch.float64)
    self.generator = torch.Generator().manual_seed(torch.initial_seed())

  def __len__(self) -> int:
    return self.history.shape[0]

  @torch.no_grad()
  def add(self, history: torch.Tensor, teacher_action: torch.Tensor) -> None:
    # 每条样本只分配一次独立随机优先级；保留最高的K条，得到历史流的
    # 均匀子集。history、动作标签和优先级始终使用相同的索引。
    history = torch.cat((self.history, history.detach().cpu()))
    teacher_action = torch.cat((self.teacher_action, teacher_action.detach().cpu()))
    count = history.shape[0] - len(self)
    priorities = torch.cat(
      (
        self.priorities,
        torch.rand(count, dtype=torch.float64, generator=self.generator),
      )
    )
    self.total_seen += count
    if history.shape[0] > self.capacity:
      ids = priorities.topk(self.capacity, sorted=False).indices
      history = history[ids]
      teacher_action = teacher_action[ids]
      priorities = priorities[ids]
    self.history = history
    self.teacher_action = teacher_action
    self.priorities = priorities

  def sample(self, count: int, device: str) -> dict[str, torch.Tensor]:
    ids = torch.randint(len(self), (count,), generator=self.generator)
    return {
      "history": self.history[ids].to(device),
      "teacher_action": self.teacher_action[ids].to(device),
    }

  def state_dict(self) -> dict:
    return {
      "capacity": self.capacity,
      "total_seen": self.total_seen,
      "history": self.history,
      "teacher_action": self.teacher_action,
      "priorities": self.priorities,
      "rng_state": self.generator.get_state(),
    }

  def load_state_dict(self, state: dict) -> None:
    self.capacity = state["capacity"]
    self.total_seen = state["total_seen"]
    self.history = state["history"].cpu()
    self.teacher_action = state["teacher_action"].cpu()
    self.priorities = state["priorities"].cpu()
    self.generator.set_state(state["rng_state"].cpu())
