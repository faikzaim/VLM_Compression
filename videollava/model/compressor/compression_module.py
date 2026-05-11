import torch

def get_average_cls_attentions(video_attentions):
    # Take the attention values for the CLS token
    cls_attentions = video_attentions[:, :, :, 0, :]
    # Find the average across all attention heads
    attentions_mean = cls_attentions.mean(dim=2)
    return attentions_mean

def get_label_set(labels):
    res = []
    batch_count, frame_count, _, _ = labels.shape
    for b in range(batch_count):
        batch_set = []
        for f in range(frame_count):
            frame_set = []
            frame = torch.nonzero(labels[b, f])
            tokens = frame.detach().cpu().tolist()
            current_label = set()

            for i in range(len(tokens)):
                current_label.add(tokens[i][1])

                if i+1 >= len(tokens) or tokens[i][0] != tokens[i+1][0]:
                    frame_set.append(current_label.copy())
                    current_label.clear()
            batch_set.append(frame_set)
        res.append(batch_set)
    return res

class TokenPruner:
    def __init__(self):
        self.labels = None

    def __call__(self, video_features, video_attentions, k, merge_labels=None):

        if len(video_attentions.shape) > 3:
            # CLS attention to itself is excluded
            attentions_mean = get_average_cls_attentions(video_attentions)[:, :, 1:]
        else:
            # In this case the necessary operations have already been done to the attention layer
            attentions_mean = video_attentions[:, :, 1:] # CLS attention to itself is excluded

        tokens_idx = attentions_mean.topk(k, dim=-1).indices  # [batch, frames, k]

        features_no_cls =  video_features[:, :, 1:, :]
        # merge_labels.shape = [batch, frames, patches_remaining, patches]
        self.labels = torch.take_along_dim(merge_labels, tokens_idx.unsqueeze(-1), dim=2)
        return torch.take_along_dim(features_no_cls, tokens_idx.unsqueeze(-1), dim=2)

    def get_label_set(self):
        return get_label_set(self.labels)

def bipartite_soft_matching(k, r):
    """ Input is k from attention , size [ batch , tokens , channels ]. """

    k = k / k.norm(dim=-1, keepdim=True)
    a, b = k[..., ::2, :], k[..., 1::2, :]
    scores = a @ b.transpose(-1, -2)
    scores[..., 0, :] = -torch.inf  # don ’t merge cls token
    node_max, node_idx = scores.max(dim=-1)
    edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
    unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
    src_idx = edge_idx[..., :r, :]  # Merged Tokens
    dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
    unm_idx = unm_idx.sort(dim=-2)[0]  # Sort cls token back to idx 0

    def merge (x):
        """ Input is of shape [ batch , tokens , channels ]. """
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_add(-2, dst_idx.expand(n, r, c), src)
        return torch.cat([unm, dst], dim=-2)


    return merge


class TokenMerger:

    def __call__(self, video_features, video_attentions, r, merge_count=30):
        r = 8
        last_merge_r = 8

        merge_iter, remainder = divmod((256 - merge_count), 8)

        if remainder > 0:
            merge_iter += 1
            last_merge_r = remainder

        b, f, p, d = video_features.shape
        s = torch.ones(b*f, p, 1, device=video_features.device, dtype=video_features.dtype)
        video_features_flattened = video_features.view(b * f, p, d)
        mean_attention = get_average_cls_attentions(video_attentions) # shape (b, f, k)
        attention_flattened = mean_attention.view(b*f, p).unsqueeze(-1) # shape (b*f, k, 1)

        self.labels = torch.eye(p).unsqueeze(0).repeat(b*f, 1, 1).to(video_features.device)

        for i in range(merge_iter):

            if i == merge_iter - 1:
                r = last_merge_r

            merge = bipartite_soft_matching(video_features_flattened, r)

            video_features_weighted = video_features_flattened * s
            attention_weighted = attention_flattened * s

            s = merge(s)
            video_features_flattened = merge(video_features_weighted)/ s
            attention_flattened = merge(attention_weighted) / s

            self.labels = merge(self.labels)

        self.labels = self.labels.view(b, f, -1, p)[:,:,1:,1:]

        video_features = video_features_flattened.view(b, f, -1, d) # shape (b, f, p-merge_iter*r, d)
        mean_attention = attention_flattened.squeeze(-1).view(b, f, -1) # shape (b, f, k-merge_iter*r)

        return video_features, mean_attention, self.labels

    def get_label_set(self):
        return get_label_set(self.labels)


class CompressorModule:
    def __init__(self, compressor_config):
        # merge_count and prune_count represent the number of tokens outputted by the Merger and the Pruner respectively
        self.merge_count = compressor_config.get('merge_count', 128)
        self.prune_count =  compressor_config.get('prune_count', 64)
        self.pruner = TokenPruner()
        self.merger = TokenMerger()

    def __call__(self, video_features, video_attentions):
        # Prune tokens with low attention
        video_features, video_attentions, merge_labels = self.merger(video_features, video_attentions, 8,
                                                                     merge_count=self.merge_count)
        video_features = self.pruner(video_features, video_attentions, self.prune_count, merge_labels=merge_labels)
        return video_features
