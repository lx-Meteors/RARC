from torch.utils.data import DataLoader, Dataset, IterableDataset
import torch

class CompressDataset(IterableDataset):
    def __init__(self, examples, batch_size):
        super(CompressDataset).__init__()
        self.examples = examples
        self.batch_size = batch_size

    def __iter__(self):
        input_ids = []
        ae_targets = []
        lm_targets = []
        lingua2 = []
        for example in self.examples:
            input_ids.append(example["inputs"])
            ae_targets.append(example["ae_target"])
            lm_targets.append(example["lm_target"])
            if len(example["lingua2"]) != 0:
                for item in example["lingua2"]:
                    lingua2.append(example["lingua2"][item])
                    lingua2.append(torch.LongTensor([128000]))
            else:
                lingua2.append(torch.LongTensor([128000]))
            lingua2 = torch.cat(lingua2, dim=0)

            if self.batch_size == len(input_ids):
                yield {"input_ids":torch.stack(input_ids),
                       "ae_targets":torch.stack(ae_targets),
                       "lm_targets":torch.stack(lm_targets),
                       "lingua2":lingua2,}
                input_ids = []
                ae_targets = []
                lm_targets = []
                lingua2 = []

                
def get_dataset(task_type, examples, batch_size):
    if task_type == "Compress":
        return CompressDataset(examples, batch_size)
    
    raise Exception("Don't exist [{task_type}] task.")