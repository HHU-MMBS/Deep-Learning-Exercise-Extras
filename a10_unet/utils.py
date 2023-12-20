from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms as T

class CityscapesDownsampled(torch.utils.data.Dataset):
    def __init__(self, img_path, label_path, transform=None, target_transform=None):
        self.ignore_index=250
        self.img_path = img_path
        self.label_path = label_path
        self.imgs = torch.load(img_path)
        self.labels = torch.load(label_path)
        self.transform = transform
        self.target_transform = target_transform
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', \
                    'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', \
                    'train', 'motorcycle', 'bicycle']
        self.class_map = dict(zip(self.valid_classes, range(len(self.valid_classes))))
        self.n_classes=len(self.valid_classes)
        self.colors = [   [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]]

        self.label_colours = dict(zip(range(self.n_classes), self.colors))
    
    def __len__(self):
        return len(self.imgs)
    
    def encode_segmap(self, mask):
        # remove unwanted classes and recitify the labels of wanted classes
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask.long().squeeze()

    def decode_segmap(self, temp):
        # convert gray scale to color
        temp = temp.numpy()
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def seg_show(self, seg):
        """
        shows an image on the screen. mean of 0 and variance of 1 will show the images unchanged in the screen
        """
        seg = self.decode_segmap(seg.squeeze_())
        plt.imshow(seg)
    
    def plot_triplet(self, img, seg, pred):
        """
        shows a triplet of: image + ground truth + predicted segmentation
        """
        plt.subplots(ncols=3, figsize=(18,10))
        plt.subplot(131)
        plt.title("Original Image")
        self.img_show(img, mean=torch.tensor([0.5]), std=torch.tensor([0.5]))
        plt.subplot(132)
        plt.title("Ground Truth")
        self.seg_show(seg)
        plt.subplot(133)
        plt.title("Predicted")
        self.seg_show(pred)
        plt.show()
    
    def img_show(self, img, mean=torch.tensor([0.0], dtype=torch.float32), std=torch.tensor([1], dtype=torch.float32)):
        """
        shows an image on the screen.
        mean of 0 and variance of 1 will show the images unchanged in the screen
        """
        unnormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
        npimg = unnormalize(img).numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    def __getitem__(self, index):
        img = self.imgs[index,...]
        seg = self.labels[index,...]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            seg = self.target_transform(seg)
        
        return img, seg
    

class CityscapesSubset(CityscapesDownsampled):
    def __init__(self, dataset, indices, **params):
        super().__init__(**params)
        self.train_dataset_test = torch.utils.data.Subset(dataset, indices)
    def __len__(self):
        return len(self.train_dataset_test)
    
    def __getitem__(self, index):
        img = self.imgs[index,...]
        seg = self.labels[index,...]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            seg = self.target_transform(seg)
        
        return img, seg

def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model {path} is loaded from epoch {checkpoint['epoch']} , loss {checkpoint['loss']}")
    return model

def plot_stats(dict_log, modelname="",baseline=None, title=None, scale_metric=100):
    plt.figure(figsize=(15,10))
    fontsize = 14
    plt.subplots_adjust(hspace=0.3)
    plt.subplot(2,1,1)
    x_axis = list(range(len(dict_log["val_metric"])))
    
    y_axis_train = [i * scale_metric for i in dict_log["train_metric"]]
    y_axis_val = [i * scale_metric for i in dict_log["val_metric"]]
    plt.plot(y_axis_train, label=f'{modelname} Train mIoU')
    plt.scatter(x_axis, y_axis_train)

    plt.plot( y_axis_val, label=f'{modelname} Validation mIoU')
    plt.scatter(x_axis, y_axis_val)

    plt.ylabel('mIoU in %')
    plt.xlabel('Number of Epochs')
    plt.title("mIoU over epochs", fontsize=fontsize)
    if baseline is not None:
        plt.axhline(y=baseline, color='red', label="Acceptable performance")
    plt.legend(fontsize=fontsize, loc='best')

    plt.subplot(2,1,2)
    plt.plot(dict_log["train_loss"] , label="Training")


    plt.scatter(x_axis, dict_log["train_loss"], )
    plt.plot(dict_log["val_loss"] , label='Validation')
    plt.scatter(x_axis, dict_log["val_loss"])

    plt.ylabel('Loss value')
    plt.xlabel('Number of Epochs')
    plt.title("Loss over epochs", fontsize=fontsize)
    plt.legend(fontsize=fontsize, loc='upper right')


def print_module_summary(module, inputs, max_nesting=3, skip_redundant=True, **kwargs):
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)
    assert isinstance(inputs, (tuple, list))

    # Register hooks.
    entries = []
    nesting = [0]
    def pre_hook(_mod, _inputs):
        nesting[0] += 1
    def post_hook(mod, _inputs, outputs):
        nesting[0] -= 1
        if nesting[0] <= max_nesting:
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [t for t in outputs if isinstance(t, torch.Tensor)]
            entries.append(EasyDict(mod=mod, outputs=outputs))
    hooks = [mod.register_forward_pre_hook(pre_hook) for mod in module.modules()]
    hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

    # Run module.
    outputs = module(*inputs, **kwargs)
    for hook in hooks:
        hook.remove()

    # Identify unique outputs, parameters, and buffers.
    tensors_seen = set()
    for e in entries:
        e.unique_params = [t for t in e.mod.parameters() if id(t) not in tensors_seen]
        e.unique_buffers = [t for t in e.mod.buffers() if id(t) not in tensors_seen]
        e.unique_outputs = [t for t in e.outputs if id(t) not in tensors_seen]
        tensors_seen |= {id(t) for t in e.unique_params + e.unique_buffers + e.unique_outputs}

    # Filter out redundant entries.
    if skip_redundant:
        entries = [e for e in entries if len(e.unique_params) or len(e.unique_buffers) or len(e.unique_outputs)]

    # Construct table.
    rows = [[type(module).__name__, 'Parameters', 'Buffers', 'Output shape', 'Datatype']]
    rows += [['---'] * len(rows[0])]
    param_total = 0
    buffer_total = 0
    submodule_names = {mod: name for name, mod in module.named_modules()}
    for e in entries:
        name = '<top-level>' if e.mod is module else submodule_names[e.mod]
        param_size = sum(t.numel() for t in e.unique_params)
        buffer_size = sum(t.numel() for t in e.unique_buffers)
        output_shapes = [str(list(t.shape)) for t in e.outputs]
        output_dtypes = [str(t.dtype).split('.')[-1] for t in e.outputs]
        rows += [[
            name + (':0' if len(e.outputs) >= 2 else ''),
            f'{param_size:,}' if param_size else '-',
            str(buffer_size) if buffer_size else '-',
            (output_shapes + ['-'])[0],
            (output_dtypes + ['-'])[0],
        ]]
        for idx in range(1, len(e.outputs)):
            rows += [[name + f':{idx}', '-', '-', output_shapes[idx], output_dtypes[idx]]]
        param_total += param_size
        buffer_total += buffer_size
    rows += [['---'] * len(rows[0])]
    rows += [['Total', f'{param_total:,}', str(buffer_total), '-', '-']]

    # Print table.
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    print()
    for row in rows:
        print('  '.join(cell + ' ' * (width - len(cell)) for cell, width in zip(row, widths)))
    print()
    return outputs


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]
