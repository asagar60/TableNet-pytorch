import torch
import torch.nn as nn

class TableNetLoss(nn.Module):
    def __init__(self):
        super(TableNetLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, table_pred, table_gt, col_pred = None, col_gt = None, ):

        table_loss = self.bce(table_pred, table_gt)
        column_loss = self.bce(col_pred, col_gt)
        return table_loss, column_loss
