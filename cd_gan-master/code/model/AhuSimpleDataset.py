import torch.utils.data as data


class AhuSimpleDataset(data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, x, y, transform=None):
        """
        Args:
            x (numpy or DataFrame): data
            y (numpy or DataFrame): label
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_item = self.x.iloc[idx].values
        y_item = self.y.iloc[idx].values

        sample = (x_item, y_item)

        if self.transform:
            x_item = self.transform(x_item)
            y_item = self.transform(y_item)
        # result = pd.concat([x_item,y_item], axis=1).values
        #         print(result.shape)
        return x_item, y_item
