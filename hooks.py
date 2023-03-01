import numpy as np
import torch
from tqdm import tqdm 
from hook import Hook

@torch.no_grad()
def extract_features_gap(model, device, data, phase='train', layers_name_list=['features'], layers_idxs = None, n_loops=1, exp_n_feat=None):
    """Extract and concatenate features maps after a global average pooling.

        Args:
        model: the CNN.
        device: torch device.
        data: container class for the pytorch dataloaders.
        phase: which dataloader (from data) to use.
        layers_name_list: the layer(s) of the CNN from where to extract the feature maps. 
                    It depends on the architectures: for vgg expects a list with one element, 
                    for inception a list of names.
        layers_idxs: indices of the layers from where to extract the feature maps. Expected for vgg, 
                    ignored for iception.
        n_loops: Number of loops over the entire dataset. 
                If 1 all data will be used. 
                For training data with augmentation you would use a larger number, e.g. 6.
                Fof validation 1 is the reccommended choice.
        exp_n_feat: expected number of features, required if not using defaults setup for vgg or inception.
        Reference: for the use of hooks https://www.kaggle.com/appian/implementing-image-feature-extraction-in-pytorch
    """
    
    model = model.to(device)
    model.eval()

    # register hooks on each layer
    if 'vgg' in model.name:
        layers_list = [model._modules[layers_name_list[0]][i] for i in layers_idxs]
        if exp_n_feat is None:
            exp_n_feat = 1472
    elif 'inception' in model.name:
        layers_list = [model._modules[ln] for ln in layers_name_list]
        if exp_n_feat is None:
            exp_n_feat = 3360
    elif 'effnet' in model.name:
        layers_list = [model._modules[layers_name_list[0]][i] for i in layers_idxs]
        if len(layers_name_list) == 2: # wether to use the head
            layers_list.append(model._modules[layers_name_list[1]])
        if exp_n_feat is None:
            exp_n_feat = 768 #464
            if len(layers_name_list) == 2:
                exp_n_feat += 1280
    else:
        raise NotImplementedError("")
    hookF = [Hook(layer) for layer in layers_list]

    loader = data.dataloaders[phase]
    classes_list = []
    path_list = []
    n_batches = len(loader)*n_loops
    for _ in range(n_loops):
        for i_batch, (inputs, classes, paths) in tqdm(enumerate(loader), total=len(loader)):
            if len(inputs.size()) == 5:
                bs, ncrops, c, h, w = inputs.size()
                _ = model(inputs.view(-1, c, h, w).to(device))
            else:
                _ = model(inputs.to(device))
            classes_list.extend(classes.numpy())
            path_list.extend(paths)
        
    # print('***'*3+'  Forward Hooks Inputs & Outputs  '+'***'*3)
    # for hook in hookF:
    #     #print(hook.input)
    #     #print(hook.input[0].shape)
    #     print(len(hook.output))
    #     print(hook.output[0].shape)
    #     print(len(hook.features))
    #     print(hook.features[0].shape)
    #     print('---'*17)

    # hookF contains len(layers) feature maps with shape (n_batches, batch_size, n_channels)
    # First we create a list of feature maps for each batch
    # fmaps_one_batch = [hook.features[-1] for hook in hookF]
    # print(f"There are {len(fmaps_one_batch)} feature maps")
    # for fmap in fmaps_one_batch:
    #    print(f"The maps have shape: {fmap.shape}")
    # # We then concatenate the feature maps (actually 1D arrays with size n_channels)
    # # In order to have a feature vector per image
    # print(np.concatenate(fmaps_one_batch, axis=1).shape)
    ## Considering all batches one further concatenation is needed.
    features = []
    for i in range(n_batches):
        hf = [hook.features[i] for hook in hookF]
        features.append(np.concatenate(hf, axis=1))
    
    features = np.concatenate(features)

    #print(features.shape[1])
    assert features.shape[1] == exp_n_feat
    assert (features.shape[0] == len(classes_list)) or (features.shape[0] == len(classes_list)*10)
    fmaps_first_batch = [hook.features[0] for hook in hookF]
    fmap_first = fmaps_first_batch[0][0]
    np.testing.assert_allclose(features[0][:len(fmap_first)], fmap_first)

    central_batch = int(len(loader)/2)
    fmaps_central_batch = [hook.features[central_batch] for hook in hookF]
    central_sample = int(len(fmaps_central_batch[0])/2)
    fmap_central = fmaps_central_batch[0][central_sample]
    np.testing.assert_allclose(
        features[central_batch*len(fmaps_central_batch[0])+central_sample][:len(fmap_central)],
        fmap_central
        )
        

    fmaps_last_batch = [hook.features[-1] for hook in hookF]
    fmap_last = fmaps_last_batch[-1][-1]
    np.testing.assert_allclose(features[-1][-len(fmap_last):], fmap_last)

    if n_loops > 1:
        central_batch = len(loader)+int(len(loader)/2)
        fmaps_central_batch = [hook.features[central_batch] for hook in hookF]
        central_sample = int(len(fmaps_central_batch[0])/2)
        fmap_central2 = fmaps_central_batch[0][central_sample]
        last_batch_length = len(fmaps_last_batch[-1])
        diff = len(fmaps_central_batch[0]) - last_batch_length
        np.testing.assert_allclose(
            features[central_batch*len(fmaps_central_batch[0])-diff+central_sample][:len(fmap_central2)],
            fmap_central2
            )
        # We want the two central feature maps for different loops to be different
        np.testing.assert_raises(AssertionError, np.testing.assert_allclose,
            fmap_central,
            fmap_central2
            )

    
    return features, classes_list, path_list

def extract_features(model, device, data, phase, layers_name_list, layers_idxs, n_loops=1, exp_n_feat=None):
    if 'vgg' in model.name or 'inception' in model.name or 'effnet' in model.name:
        return extract_features_gap(model, device, data, phase, layers_name_list, layers_idxs, n_loops, exp_n_feat)
    else:
        raise NotImplementedError()
