import logging

import torch
import torch.nn as nn


class MultiResolutionHashEncoder(nn.Module):
    """
        I write separate forward functions for 2D and 3D because I can't put if statements inside the forward
        pass, otherwise torch.compile will throw all sorts of errors.
    """
    def __init__(self,
                 compile: bool = False,
                 **kwargs):
        super().__init__()
        logger = logging.getLogger("__main__")

        self.PI1: int = 1
        self.PI2: int = 2654435761
        self.PI3: int = 805459861

        self.compile = compile
        self.D = kwargs.get("D", 3)
        self.L_H = kwargs.get("L_H", 2)
        self.L_NH = kwargs.get("L_NH", 6)
        self.polarity = kwargs.get("polarity", False)
        self.feature_size = kwargs.get("feature_size", 4)
        self.resolutions = kwargs.get("resolutions", None)
        self.levels = kwargs.get("levels", self.L_H + self.L_NH) # Should be equal to L_H + L_NH
        self.log2_entries_per_level = kwargs.get("log2_entries_per_level", 19)

        try:
            self.index = getattr(self, f"index{self.D}d")
            self.forward = getattr(self, f"forward{'_pol' if self.polarity else '_nopol'}")
        except AttributeError:
            raise ValueError(f"Invalid number of dimensions: {self.D}")
        logger.info(f"Using {self.D}D Multi-Resolution Hash Encoder with {self.L_H} hashed levels and {self.L_NH} non-hashed levels and polarity: {self.polarity}")

        def get_hashmap():
            with torch.no_grad():
                hashmap = torch.zeros((self.levels, 1 << self.log2_entries_per_level, self.feature_size), dtype=torch.float32)
                hashmap.uniform_(-1e-4, 1e-4)
                hashmap = nn.Parameter(hashmap) # L x T x F where T = 2^log2_entries_per_level, F = feature_size
            return hashmap

        # build the hash tables
        if not self.polarity:
            self.hashmap = get_hashmap()
        else:
            self.hashmap_neg = get_hashmap()
            self.hashmap_pos = get_hashmap()

    def hash_linear_congruential3d(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, log2_entries_per_level: int) -> torch.Tensor:
        return (x * self.PI1 ^ y * self.PI2 ^ t * self.PI3) % (1 << log2_entries_per_level)
    
    def hash_linear_congruential2d(self, x: torch.Tensor, y: torch.Tensor, log2_entries_per_level: int) -> torch.Tensor:
        return (x * self.PI1 ^ y * self.PI2) % (1 << log2_entries_per_level)
    
    def index3d(self, eventBlock: torch.Tensor) -> torch.Tensor:
        x, y, t = eventBlock[:, :, 0], eventBlock[:, :, 1], eventBlock[:, :, 2]

        #! Asserts don't work with torch.compile
        if not self.compile:
            assert x.shape == y.shape == t.shape, f"x.shape: {x.shape}, y.shape: {y.shape}, t.shape: {t.shape}"
            assert x.min() >= 0 and x.max() <= 1, f"x coordinate should be in [0, 1]"
            assert y.min() >= 0 and y.max() <= 1, f"y coordinate should be in [0, 1]"
            assert t.min() >= 0, f"t should be non-negative"

        scaled_x = x.unsqueeze(-1) * self.resolutions[:, 0] # B x N x L
        scaled_y = y.unsqueeze(-1) * self.resolutions[:, 1] # B x N x L
        scaled_t = t.unsqueeze(-1) * self.resolutions[:, 2] # B x N x L

        floor_scaled_x = scaled_x.int() # B x N x L
        floor_scaled_y = scaled_y.int() # B x N x L
        floor_scaled_t = scaled_t.int() # B x N x L

        ceil_scaled_x = torch.min(floor_scaled_x + 1, self.resolutions[:, 0][None, None, :]) # B x N x L
        ceil_scaled_y = torch.min(floor_scaled_y + 1, self.resolutions[:, 1][None, None, :]) # B x N x L
        ceil_scaled_t = torch.min(floor_scaled_t + 1, self.resolutions[:, 2][None, None, :]) # B x N x L

        # all combinations of the 8 corners of the cube B X N x L x 8 x 3
        corners = torch.stack([
            torch.stack([floor_scaled_x, floor_scaled_y, floor_scaled_t], dim=-1),
            torch.stack([floor_scaled_x, floor_scaled_y, ceil_scaled_t], dim=-1),
            torch.stack([floor_scaled_x, ceil_scaled_y, floor_scaled_t], dim=-1),
            torch.stack([floor_scaled_x, ceil_scaled_y, ceil_scaled_t], dim=-1),
            torch.stack([ceil_scaled_x, floor_scaled_y, floor_scaled_t], dim=-1),
            torch.stack([ceil_scaled_x, floor_scaled_y, ceil_scaled_t], dim=-1),
            torch.stack([ceil_scaled_x, ceil_scaled_y, floor_scaled_t], dim=-1),
            torch.stack([ceil_scaled_x, ceil_scaled_y, ceil_scaled_t], dim=-1)
        ], dim=-2).int()  # B x N x L x 8 x 3

        # calculate the weights for each corner B x N x L x 8
        weights = torch.prod(1 - (corners - torch.stack([scaled_x, scaled_y, scaled_t], dim=-1).unsqueeze(-2)).abs(), dim=-1)

        # Calculate the indices for the hash table (Hash + Non-hash levels depending on the resolution)
        # B x N x L_H x 8 where L_H is the number of levels that need hashing
        hash_values = self.hash_linear_congruential3d(corners[:,:,-self.L_H:,:,0],
                                                      corners[:,:,-self.L_H:,:,1],
                                                      corners[:,:,-self.L_H:,:,2],
                                                      self.log2_entries_per_level)
        # B x N x L_NH x 8 where L_NH is the number of levels that don't need hashing
        nonhash_values = corners[:,:,:self.L_NH,:,0] +\
                         corners[:,:,:self.L_NH,:,1] * self.resolutions[:self.L_NH,0][None,None,:,None] +\
                         corners[:,:,:self.L_NH,:,2] * (self.resolutions[:self.L_NH,0] * self.resolutions[:self.L_NH,1])[None,None,:,None]
        
        return hash_values, nonhash_values, weights
    
    def index2d(self, eventBlock: torch.Tensor) -> torch.Tensor:
        x, y = eventBlock[:, :, 0], eventBlock[:, :, 1]

        #! Asserts don't work with torch.compile
        if not self.compile:
            assert x.shape == y.shape, f"x.shape: {x.shape}, y.shape: {y.shape}"
            assert x.min() >= 0 and x.max() <= 1, f"x coordinate should be in [0, 1]"
            assert y.min() >= 0 and y.max() <= 1, f"y coordinate should be in [0, 1]"
        
        scaled_x = x.unsqueeze(-1) * self.resolutions[:, 0]
        scaled_y = y.unsqueeze(-1) * self.resolutions[:, 1]

        floor_scaled_x = scaled_x.int()
        floor_scaled_y = scaled_y.int()

        ceil_scaled_x = torch.min(floor_scaled_x + 1, self.resolutions[:, 0][None, None, :])
        ceil_scaled_y = torch.min(floor_scaled_y + 1, self.resolutions[:, 1][None, None, :])

        corners = torch.stack([
            torch.stack([floor_scaled_x, floor_scaled_y], dim=-1),
            torch.stack([floor_scaled_x, ceil_scaled_y], dim=-1),
            torch.stack([ceil_scaled_x, floor_scaled_y], dim=-1),
            torch.stack([ceil_scaled_x, ceil_scaled_y], dim=-1)
        ], dim=-2).int()

        weights = torch.prod(1 - (corners - torch.stack([scaled_x, scaled_y], dim=-1).unsqueeze(-2)).abs(), dim=-1)

        hash_values = self.hash_linear_congruential2d(corners[:,:,-self.L_H:,:,0],
                                                      corners[:,:,-self.L_H:,:,1],
                                                      self.log2_entries_per_level)
        nonhash_values = corners[:,:,:self.L_NH,:,0] +\
                         corners[:,:,:self.L_NH,:,1] * self.resolutions[:self.L_NH,0][None,None,:,None]

        return hash_values, nonhash_values, weights

    def forward_pol(self, eventBlock: torch.Tensor) -> torch.Tensor:
        """
            Forward pass of the Multi-Resolution Hash Encoder for 4D events (i.e. polarities).


            Args:
                x: x coordinate of the event rescaled to [0, 1]. (B,N)
                y: y coordinate of the event rescaled to [0, 1]. (B,N)
                t: time bin of the event. (B,N)
                (Say if each bucket is 20us, and there's a total of 1000 buckets, then t is in [0, 1000])
                p: polarity of the event. (B,N)

                (x,y,t,p) or (x,y,p)

            Returns:
                The feature field of the events. (B, N, L*F) where L is the number of levels and F is the feature size.
        """
        p = eventBlock[:, :, -1].reshape(-1)
        B, N = eventBlock.shape[0], eventBlock.shape[1]
        hash_values, nonhash_values, weights = self.index(eventBlock)
        
        hash_values, nonhash_values = hash_values.reshape(B * N, self.L_H, 2**self.D), nonhash_values.reshape(B * N, self.L_NH, 2**self.D)
        neg_idx, pos_idx = (p == 0).nonzero().squeeze(), (p == 1).nonzero().squeeze() #! Boolean indexing doesn't seem to work with torch.compile
        hashmap_features = torch.zeros((B * N, self.levels, 2**self.D, self.feature_size), dtype=self.hashmap_neg.dtype, device=eventBlock.device)
        for i in range(self.L_NH):
            hashmap_features[neg_idx, i, :, :] = self.hashmap_neg[i][nonhash_values[neg_idx, i, :]]
            hashmap_features[pos_idx, i, :, :] = self.hashmap_pos[i][nonhash_values[pos_idx, i, :]]
        for i in range(self.L_H):
            hashmap_features[neg_idx, i + self.L_NH, :, :] = self.hashmap_neg[i + self.L_NH][hash_values[neg_idx, i, :]]
            hashmap_features[pos_idx, i + self.L_NH, :, :] = self.hashmap_pos[i + self.L_NH][hash_values[pos_idx, i, :]]
        hashmap_features = hashmap_features.reshape(B, N, self.levels, 2**self.D, self.feature_size)

        interpolated_features = torch.sum(weights.unsqueeze(-1) * hashmap_features, dim=-2)
        interpolated_features = interpolated_features.reshape(B, N, -1)
        return interpolated_features

    def forward_nopol(self, eventBlock: torch.Tensor) -> torch.Tensor:
        """
            Forward pass of the Multi-Resolution Hash Encoder.

            Args:
                x: x coordinate of the event rescaled to [0, 1]. (B,N)
                y: y coordinate of the event rescaled to [0, 1]. (B,N)
                t: time bin of the event. (B,N)
                (Say if each bucket is 20us, and there's a total of 1000 buckets, then t is in [0, 1000])

                (x,y,t) or (x,y)

            Returns:
                The feature field of the events. (B, N, L*F) where L is the number of levels and F is the feature size.
        """
        B, N = eventBlock.shape[0], eventBlock.shape[1] # Batch size, Number of events in each batch
        hash_values, nonhash_values, weights = self.index(eventBlock)
        
        # We have a hashmap of size L x T x F where T = 2^log2_entries_per_level, F = feature_size
        # We want to index it with N x L x 8 index tensor. That is, each row in the 1st dim of the index matrix,
        # index into the rows of the hashmap tensor. The 8 corners index into the 1st dim of the hashmap tensor.
        # We don't want to use gather with expand, because in backward pass it goes OOM. Well I don't like loops,
        # but I couldn't find a cleverer way to use gather which doesn't either expand on the N or F dimension.
        hashmap_features = torch.zeros((B, N, self.levels, 2**self.D, self.feature_size), dtype=self.hashmap.dtype, device=eventBlock.device)
        for i in range(self.L_NH):
            hashmap_features[:, :, i, :, :] = self.hashmap[i][nonhash_values[:, :, i, :]]
        for i in range(self.L_H):
            hashmap_features[:, :, i + self.L_NH, :, :] = self.hashmap[i + self.L_NH][hash_values[:, :, i, :]] 

        interpolated_features = torch.sum(weights.unsqueeze(-1) * hashmap_features, dim=-2) # B x N x L x F
        interpolated_features = interpolated_features.reshape(B, N, -1) # B x N x (L*F)
        return interpolated_features
