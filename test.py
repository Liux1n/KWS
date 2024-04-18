import torch
import unittest
import numpy as np
# from utils import Buffer

def task_average_forgetting(acc_matrix):
    # Number of tasks (not including the final model evaluation)
    num_tasks = acc_matrix.shape[0]

    # Initialize the forgetting list
    forgetting = []

    # Calculate the maximum accuracy for each task from stages before the last and the last accuracy
    for task_index in range(num_tasks):
        # Filter out zero values but include the last value in case it's the only record
        filtered_acc = acc_matrix[task_index][acc_matrix[task_index] != 0]
        if len(filtered_acc) > 1:
            max_acc_before_last = np.max(filtered_acc[:-1])  # Maximum before the last training
            last_acc = filtered_acc[-1]  # The last training result
            forgetting.append(max_acc_before_last - last_acc)
        elif len(filtered_acc) == 1:  # Only one non-zero entry, no forgetting possible
            forgetting.append(0)

    # Calculate average forgetting
    if len(forgetting) > 0:
        average_forgetting = np.mean(forgetting)
    else:
        average_forgetting = 0  # In case all entries were zero and filtered out

    return average_forgetting / 100

    
class Buffer_NRS:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, batch_size,device):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.num_seen_examples = 0
        self.buffer = {}
        self.attributes = ['examples', 'labels']
        
        self.buffer['examples'] = torch.empty((self.buffer_size, 1, 49, 10), device=self.device)
        self.buffer['labels'] = torch.empty((self.buffer_size), device=self.device)
        print("Buffer initialized")
    
    def naive_reservoir(self) -> int:
        """
        Naive Reservoir Sampling algorithm.

        """
        if self.num_seen_examples < self.buffer_size:
            return self.num_seen_examples

        rand = np.random.randint(0, self.num_seen_examples + 1)
        if rand < self.buffer_size:
            return rand
        else:
            return -1

    def add_data(self, examples, labels):
        """
        Add data to the buffer.
        examples: torch.Size([128, 1, 49, 10])
        labels: torch.Size([128])
        """
        # if buffer is not full, add batch data to buffer
        if self.num_seen_examples < self.buffer_size:
            self.buffer['examples'][self.num_seen_examples:self.num_seen_examples + self.batch_size] = examples
            self.buffer['labels'][self.num_seen_examples:self.num_seen_examples + self.batch_size] = labels
            self.num_seen_examples += self.batch_size
            # print("Data added to buffer")
        else:
            for i in range(self.batch_size):
                sample_index = self.naive_reservoir(self.num_seen_examples, self.buffer_size)
                if sample_index != -1:
                    self.buffer['examples'][sample_index] = examples[i]
                    self.buffer['labels'][sample_index] = labels[i]
                    # print("Data added to buffer")
                    self.num_seen_examples += 1

    def get_data(self):
        """
        Get data from the buffer.
        """
        indices = torch.randperm(self.num_seen_examples)[:self.batch_size]
        return self.buffer['examples'][indices], self.buffer['labels'][indices]

    def is_empty(self):
        """
        Check if the buffer is empty.
        """
        return self.num_seen_examples == 0  
    
    def get_class_count(self):
        """
        Get the number of examples for each class in the buffer.
        """
        class_count = {}
        for label in self.buffer['labels']:
            if label.item() in class_count:
                class_count[label.item()] += 1
            else:
                class_count[label.item()] = 1
        return class_count


        

        



class TestBuffer(unittest.TestCase):
    def setUp(self):
        self.buffer_size = 1024
        self.device = torch.device('cpu')
        self.buffer = Buffer_NRS(self.buffer_size,128, self.device)
        print("Setup completed")

    def test_add_and_get_data(self):
        # examples  torch.Size([1024, 1, 49, 10])
        # label  torch.Size([1024])

        batch_size = 128    
        examples = torch.rand([batch_size, 1, 49, 10])

        # labels is random string from 'a' to 'z'
        labels = torch.randint(0, 26, (batch_size,))
        print(labels)


        
        
        
        self.buffer.add_data(examples, labels)
        print("Data added to buffer")


        out_examples, out_labels = self.buffer.get_data()
        print("Examples:", out_examples, "Labels:", out_labels)

     


if __name__ == '__main__':
    # unittest.main()
    # Provided accuracy matrix
    acc_matrix = np.array([
        [71.32000661, 0., 0., 0.],
        [69.72575582, 65.62861391, 0., 0.],
        [70.66743763, 65.87642491, 68.98232282, 0.],
        [70.21311746, 65.53774988, 68.66842888, 64.97604494]
    ])

    # Calculate and print the average forgetting
    average_forgetting = task_average_forgetting(acc_matrix)
    print(average_forgetting)