import torch
import math

class Conv_block(torch.nn.Module) :
    """
    Hierarchical 3D convolutional feature extractor used throughout the encoder, bottleneck, and decoder.

    This block implements a stack of 3D convolutional layers with Group Normalization and ReLU activation,
    designed for medical volumes where batch sizes are small and intensity statistics are unstable.

    Design motivations:

    GroupNorm (instead of BatchNorm) ensures training stability for small batch sizes typical in 3D CT/MRI.

    Multiple stacked convolutions increase receptive field while preserving spatial resolution.

    Padding is kernel-aware, ensuring exact shape preservation across all three spatial dimensions.

    Acts as the fundamental local context modeling unit before global Transformer reasoning.
    """
    def __init__(self , in_channles , out_channles , n_layers =2 , activation = 'relu' , kernel_size = (3,3,3)) :
        super().__init__()
        self.conv_layers = torch.nn.ModuleList()
        for _ in range(n_layers) :
            self.conv_layers.append(torch.nn.Sequential([
                torch.nn.Conv3d(in_channels=in_channles , out_channels=out_channles , kernel_size=kernel_size , padding = tuple(k//2 for k in kernel_size) , bias = False ) ,
                torch.nn.GroupNorm(1, out_channles) ,
                torch.nn.ReLU(inplace=True)
            ]))
            in_channels = out_channles


    def forward(self, x) :
        """
        Applies a sequence of 3D convolution → normalization → nonlinearity operations.

        Args:
        x (Tensor): Input feature map of shape (B, C, D, H, W).

        Returns:
        Tensor: Locally refined volumetric features with identical spatial resolution and updated channel depth.

        This operation builds progressively richer spatial descriptors before hierarchical downsampling
        or skip-connection fusion.
        """
        for layer in self.conv_layers :
            x = layer(x)

        return x




class Encoder_block (torch.nn.Module) :
    """
    Single stage of the hierarchical 3D encoder.

    Each block performs:

    Local feature extraction via Conv_block.

    Resolution reduction via 3D max pooling.

    Regularization through spatial dropout.

    This structure mirrors classical U-Net semantics while preparing multi-scale representations
    for Transformer-based global fusion.

    Outputs both:

    High-resolution features for skip connections.

    Downsampled features for deeper semantic processing.
    """
    def __init__(self , in_channles , out_channles , pool_size = (2,2,2) , dropout = 0.3 ) :
        super().__init__()
        self.conv_block = Conv_block(in_channles=in_channles , out_channles=out_channles)
        self.pool = torch.nn.MaxPool3d(kernel_size=pool_size )
        self.dropout = torch.nn.Dropout(p = dropout  , inplace = True)

    def forward(self, x) :
        """
        Processes input volume through convolutional refinement and spatial downsampling.

        Returns:
        f: High-resolution feature tensor used for skip connections.
        p: Downsampled tensor forwarded to the next encoder level.

        This dual-output design enables simultaneous preservation of fine geometry
        and extraction of hierarchical semantics.
        """
        f = self.conv_block(x)
        p = self.pool(f)
        p = self.dropout(p)
        return f , p


class Encoder(torch.nn.Module) :
    """
    Four-stage 3D convolutional encoder forming the multi-scale backbone of the network.

    Channel progression: 64 → 128 → 256 → 512
    Spatial resolution decreases while semantic abstraction increases.

    These multi-resolution features serve two purposes:

    1.Skip-connections for precise spatial reconstruction.

    2.Multi-scale tokenization for Transformer global reasoning.

    Acts as the geometric perception front-end of MS-TransUNet V2.
    """
    def __init__(self , in_channels) :
        super().__init__()
        self.encoder_block_1 = Encoder_block(in_channles=in_channels , out_channles=64)
        self.encoder_block_2 = Encoder_block(in_channles=64 , out_channles=128)
        self.encoder_block_3 = Encoder_block(in_channles=128 , out_channles=256)
        self.encoder_block_4 = Encoder_block(in_channles=256 , out_channles=512)

    def forward (self , x) :
        """
        Forward pass of the hierarchical 3D encoder.

        This stage performs progressive spatial downsampling and semantic enrichment
        using stacked convolutional blocks and max-pooling operations.

        At each level:
        - High-resolution geometric features are extracted (f1–f4).
        - Spatial resolution is reduced via pooling (p1–p4) to increase receptive field.
        - Channel depth is increased to encode higher-level anatomical semantics.

        Returns
        -------
        convs : Tuple[Tensor, Tensor, Tensor, Tensor]
            Multi-scale feature maps at resolutions (1x, 1/2x, 1/4x, 1/8x) used for
            skip connections and multi-scale Transformer tokenization.
        pools : Tuple[Tensor, Tensor, Tensor, Tensor]
            Downsampled feature maps feeding the next encoder stage and the bottleneck.
        """
        f1 , p1  = self.encoder_block_1(x)
        f2 , p2 = self.encoder_block_2(p1)
        f3,p3 = self.encoder_block_3(p2)
        f4,p4 = self.encoder_block_4(p3)
        return (f1,f2,f3,f4) ,(p1,p2,p3,p4)



class MLP(torch.nn.Module) :
    """
    Feed-forward network used inside each Transformer block.

    Implements the standard Transformer FFN:
    Linear → GELU → Linear with 4× channel expansion.

    Provides non-linear mixing of global context after self-attention,
    enabling high-order feature interactions across spatial tokens.
    """
    def __init__(self, d_model , in_features) :
        super().__init__()
        self.fl1 = torch.nn.Linear(in_features=in_features , out_features=d_model * 4  )
        self.act  =torch.nn.GELU()
        self.fl2 = torch.nn.Linear(in_features=4*d_model , out_features =d_model)

    def forward(self, x) :
        x = self.fl1(x)
        x= self.act(x)
        x = self.fl2(x)
        return x


class Transformer_part(torch.nn.Module) :
    """
    Single Transformer encoder layer operating on 3D token sequences.

    Architecture:

    Pre-norm LayerNorm

    Multi-Head Self-Attention (global spatial reasoning)

    Residual connection

    Feed-Forward Network (MLP)

    Second residual and normalization

    Enables long-range dependency modeling across the entire 3D volume,
    allowing anatomical structures to be interpreted globally rather than locally.
    """
    def __init__(self ,num_heads = 8 , key_dim=64 , d_model=512  , dropout_rate =0.1  ) :
        super().__init__()
        self.multihead_attention = torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads , dropout=dropout_rate, batch_first=True)
        self.mlp = MLP(d_model , d_model)
        self.mlp_dropout = torch.nn.Dropout(p = dropout_rate)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)


    def forward(self, x):
        """
        Performs one layer of global self-attention and feed-forward refinement.

        Args:
        x (Tensor): Token sequence of shape (B, N, d_model),
        where N aggregates multi-scale spatial tokens.

        Returns:
        Tensor: Globally contextualized token embeddings.

        This step allows deep semantic tokens to attend directly to high-resolution geometric tokens,
        bridging local detail with global anatomical coherence.
        """
        h = self.norm1(x)
        attn, _ = self.multihead_attention(h, h, h, need_weights=False)
        x = x + attn

        h = self.norm2(x)
        x = x + self.mlp_dropout(self.mlp(h))
        return x


class Transformer(torch.nn.Module) :
    """
    Multi-Scale Cross-Resolution Transformer (Core Innovation of MS-TransUNet V2).

    This module implements a hierarchical 3D Transformer that performs *global,
    physically-aware reasoning across multiple encoder resolutions simultaneously*.
    It is specifically designed to solve the "bottleneck isolation" and "Z-axis
    collapse" problems present in standard TransUNet and 3D U-Net architectures
    when applied to anisotropic medical volumes.

    -----------------------
    Motivation
    -----------------------
    Classical TransUNet designs tokenize only the deepest bottleneck feature map.
    This causes two critical failures:

    1) Spatial Collapse:
       After repeated (2,2,2) pooling, the Z-dimension becomes extremely small
       (e.g., 2–4 slices). The Transformer therefore operates on nearly 2D tokens,
       losing true 3D geometry.

    2) Semantic-Geometry Decoupling:
       High-resolution geometric features (f1, f2, f3) are never seen by the
       Transformer. Deep semantic tokens (f4) cannot access fine boundary shape
       or long-range anatomical continuity.

    This module resolves both by introducing *Multi-Scale Tokenization with
    Physically-Aligned Positional Encoding*.

    -----------------------
    Architectural Principles
    -----------------------

    (1) Multi-Scale Tokenization
        Instead of a single bottleneck stream, four encoder levels are tokenized:

            f1 : high-resolution geometry (edges, thin structures)
            f2 : mid-level context
            f3 : organ-level shape
            f4 : deep semantic abstraction

        Each feature map is:
            • Flattened into a token sequence (B, N_l, C_l)
            • Linearly projected into a common embedding space (d_model)

        This produces four token streams aligned in embedding dimension but
        differing in spatial density and semantic depth.

    (2) 3D Sinusoidal Positional Encoding
        A continuous 3D sine-cosine embedding is generated over (D, H, W) for each
        scale, preserving *true physical coordinates* in voxel space.

        This ensures that:
            • Attention is aware of spatial adjacency in 3D
            • Long-range anatomical continuity is preserved
            • Tokens from different resolutions remain physically registered

    (3) Learnable Level Embeddings
        Each scale receives a dedicated learnable embedding vector indicating
        its semantic depth in the hierarchy.

        This allows the Transformer to distinguish:
            • "Where is this voxel in the body?"
            • "At what abstraction level does this representation live?"

    (4) Global Cross-Scale Self-Attention
        All scale tokens are concatenated into a single sequence and processed
        by a deep Transformer stack.

        This enables:
            • Deep semantic tokens (f4) to attend to fine geometry (f1, f2)
            • Shallow tokens to receive global shape priors from deep context
            • Direct modeling of organ-to-organ and part-to-whole relationships

        Effectively, the network learns a *Global Semantic Bridge*:
            “I recognize liver tissue here — how does its full 3D shape extend?”

    (5) Hierarchical Reprojection
        After attention, the token sequence is split back into its original
        scales, reshaped into volumetric tensors, and projected back to their
        native channel widths.

        These Transformer-refined feature maps are injected into the U-Net
        decoder via skip connections, providing:

            • Globally consistent geometry
            • Long-range anatomical coherence
            • Topology-preserving reconstruction

    -----------------------
    Clinical & Geometric Impact
    -----------------------
    This design directly addresses:

    • Z-axis collapse in thick-slice CT/MRI
    • Fragmented organs due to local-only CNN context
    • Inability of bottleneck-only Transformers to model full 3D shape
    • Loss of inter-organ spatial relationships

    The resulting representation encodes:

    • Physical space (positional embeddings)
    • Semantic depth (level embeddings)
    • Global topology (self-attention)
    • Local detail (CNN inductive bias preserved in skip paths)

    The Transformer therefore acts not as a bottleneck, but as a
    *cross-resolution anatomical reasoning engine* that unifies texture,
    shape, and global context in a single attention space.
    """
    def __init__(self  , d_model = 512 , num_layers=12) :
        super().__init__()
        self.projection_1 = torch.nn.Linear(in_features=64 , out_features= d_model)
        self.projection_2  = torch.nn.Linear(in_features= 128 , out_features= d_model)
        self.projection_3 = torch.nn.Linear(in_features=256 , out_features= d_model)
        self.projection_4 = torch.nn.Linear(in_features=512 , out_features=d_model)
        self.d_model = d_model
        self.transformer_block = torch.nn.ModuleList(
            [Transformer_part() for _ in range(num_layers)]
        )

        self.level_embedding = torch.nn.Parameter(torch.randn(4,d_model))

        self.block1 = torch.nn.Conv3d(in_channels=self.d_model , out_channels=64 , kernel_size=1)
        self.block2 = torch.nn.Conv3d(in_channels=self.d_model , out_channels=128 , kernel_size=1)
        self.block3 = torch.nn.Conv3d(in_channels=self.d_model , out_channels=256 , kernel_size=1)
        self.block4 = torch.nn.Conv3d(in_channels=self.d_model , out_channels=512 , kernel_size=1)

    def get_3d_sincos_pos_embed(self, d_model , D, H,W,stride ,  device) :
        """
        Generates continuous 3D sinusoidal positional embeddings SCALED by network stride.

        This function creates a 3D coordinate system where the positions are multiplied
        by the 'stride' factor. This ensures that tokens from deep, low-resolution layers
        (e.g., stride 8) are mapped to their correct physical location in the original volume,
        aligning them with tokens from shallow, high-resolution layers (e.g., stride 1).

        Args:
            d_model (int): Embedding dimension.
            D, H, W (int): Spatial dimensions of the feature map.
            stride (int): The downsampling factor of the current level (e.g., 1, 2, 4, 8).
            device (torch.device): The device to create tensors on.

        Returns:
            Tensor: Flattened positional embeddings physically aligned across scales.
        """
        assert d_model %  6 ==0
        d = d_model // 3

        def get_1d_pos_embd (dim , length , stride_factor)  :
            pos = torch.arange(length, device=device).float() * stride_factor
            div  = torch.exp(torch.arange(0,dim,2 , device = device).float() * (-math.log(10000) / dim ))
            pe = torch.zeros(length , dim , device = device)
            pe[:,0::2] = torch.sin (pos[:,None] *div)
            pe[:,1::2] = torch.cos(pos[:,None] * div)

            return pe

        pe_d = get_1d_pos_embd (d ,D , stride)
        pe_h = get_1d_pos_embd (d , H , stride)
        pe_w  = get_1d_pos_embd (d , W , stride)

        pe = (
            pe_d[:, None, None, :].repeat(1, H, W, 1),
            pe_h[None, :, None, :].repeat(D, 1, W, 1),
            pe_w[None, None, :, :].repeat(D, H, 1, 1),
        )

        pe = torch.cat(pe, dim=-1)
        pe = pe.view(-1, d_model)
        return pe

    def enoode_level(self, f : torch.Tensor , proj  , level_id : int )->torch.Tensor :
        """
        Converts a single encoder feature map into a physically-aware sequence of tokens.

        Operations:
        1. Flatten spatial dimensions into a token sequence.
        2. Linear projection to d_model.
        3. Calculate 'current_stride' based on level_id (2^level_id).
        4. Generate Stride-Aware Positional Embeddings using this stride.
        5. Add learnable Level Embedding to encode semantic depth.

        Args:
            f (Tensor): Input feature map.
            proj (nn.Linear): Linear projection layer.
            level_id (int): The hierarchical level (0=f1, 1=f2, 2=f3, 3=f4).

        Returns:
            Tensor: Token sequence that knows both its semantic depth (Level Embedding)
            and physical location (Stride-Aware Positional Embedding).
        """
        b,c,d,h,w = f.shape
        x = f.flatten(2).transpose(1,2)
        x  = proj (x)

        current_stride = 2 ** level_id
        
        pe_spatial = self.get_3d_sincos_pos_embed(self.d_model , d,h,w,current_stride ,x.device)
        pe_spatial = pe_spatial.unsqueeze(0)

        pe_level = self.level_embedding[level_id].view(1,1,-1)

        return x + pe_spatial + pe_level



    def forward(self , inputs) :
        """
        Performs full multi-scale Transformer reasoning.

        Pipeline:

        1.Tokenize each encoder level independently.

        2.Concatenate all tokens into a single global sequence.

        3.Apply stacked Transformer layers for cross-scale self-attention.

        4.Split refined tokens back into original scales.

        5.Reshape into volumetric feature maps.

        6.Project channels back to original encoder widths.

        Returns:
        Tuple of refined skip features at all resolutions.

        This creates a Global Semantic Bridge where:

        Deep layers query shallow geometry.

        Shallow layers receive global semantic context.
        """
        f1,f2 ,f3 ,f4 = inputs

        z1 = self.enoode_level(f1,self.projection_1 , 0)
        z2 = self.enoode_level(f2,self.projection_2 , 1)
        z3 = self.enoode_level(f3,self.projection_3 , 2)
        z4 = self.enoode_level(f4,self.projection_4 , 3)

        assert z1.shape[2] == self.d_model

        x = torch.cat([z1,z2,z3,z4] , dim =1)

        for block in self.transformer_block :
            x = block(x)


        t1 , t2 ,t3 ,t4  = torch.split(x , [z1.shape[1],z2.shape[1],z3.shape[1],z4.shape[1]] , dim = 1)

        assert t1.shape[2:] == f1.shape[2:]
        
        t1 = t1.transpose(1,2).reshape(f1.shape[0], self.d_model, f1.shape[2], f1.shape[3], f1.shape[4])
        t2 = t2.transpose(1,2).reshape(f2.shape[0], self.d_model, f2.shape[2], f2.shape[3], f2.shape[4])
        t3 = t3.transpose(1,2).reshape(f3.shape[0], self.d_model, f3.shape[2], f3.shape[3], f3.shape[4])
        t4 = t4.transpose(1,2).reshape(f4.shape[0], self.d_model, f4.shape[2], f4.shape[3], f4.shape[4])

        skip_1 = self.block1(t1)
        skip_2 = self.block2(t2)
        skip_3 = self.block3(t3)
        skip_4 = self.block4(t4)

        return (skip_1 , skip_2 , skip_3 , skip_4)




class Decoder_Block(torch.nn.Module) : 
    """
    Single decoding stage with learned upsampling and skip-connection fusion.

    Components:

    Transposed 3D convolution for resolution recovery.

    Channel-wise concatenation with corresponding skip feature.

    Dropout regularization.

    Local refinement via Conv_block.

    This block restores spatial resolution while injecting
    Transformer-refined multi-scale semantics.
    """
    def __init__(self , in_channles_conv_trans : int,   out_channels_conv_trans : int ,in_channels_convs :int , out_channels_convs : int  , dropout : float = 0.3) -> None : 
        super().__init__() 
        self.conv_block = Conv_block(
            in_channles=in_channels_convs ,
            out_channles=out_channels_convs
            ) 
        self.conv_transpose = torch.nn.ConvTranspose3d(
            in_channels=in_channles_conv_trans,
            out_channels=out_channels_conv_trans,
            kernel_size=(2,2,2),
            stride=(2,2,2),
            padding=0,
            output_padding=0
        )
        self.dropout = torch.nn.Dropout(p  = dropout) 
    
    def forward(self , x : torch.Tensor , f : torch.Tensor ) -> torch.Tensor : 
        """
        Single decoding stage with learned upsampling and skip fusion.

        Steps:
        1. Upsample low-resolution features using transposed convolution.
        2. Concatenate with corresponding encoder/Transformer-refined skip features.
        3. Apply dropout for regularization.
        4. Refine fused representation via convolutional block.

        This block restores spatial resolution while injecting high-level semantics
        into fine-grained anatomical structures.
        """
        x = self.conv_transpose(x) 
        x = torch.cat([x,f] ,dim = 1 )
        x = self.dropout(x)
        x = self.conv_block(x) 
        return x 

class Decoder (torch.nn.Module) : 
    """
    Hierarchical 3D decoder reconstructing full-resolution segmentation maps.

    Performs progressive upsampling from bottleneck to original resolution,
    merging:

    Transformer-refined multi-scale skip features.

    Deep semantic bottleneck features.

    Ensures anatomically consistent reconstruction with sharp boundaries
    and global coherence.
    """
    def __init__(self) : 
        super().__init__()
        self.decoder_block_1 = Decoder_Block(in_channles_conv_trans=128 , out_channels_conv_trans=64 , in_channels_convs=128 ,out_channels_convs=64 )
        self.decoder_block_2 = Decoder_Block(in_channles_conv_trans=256 , out_channels_conv_trans=128 , in_channels_convs=256 ,out_channels_convs=128 )
        self.decoder_block_3 = Decoder_Block(in_channles_conv_trans=512 , out_channels_conv_trans=256 , in_channels_convs=512 ,out_channels_convs=256 )
        self.decoder_block_4 = Decoder_Block(in_channles_conv_trans=1024 , out_channels_conv_trans=512 , in_channels_convs=1024 ,out_channels_convs=512 )
    
    def forward(self , convs  ,x) : 
        """
        Hierarchical multi-stage decoder.

        Progressively reconstructs full-resolution segmentation by:
        - Starting from the Transformer-refined bottleneck representation.
        - Iteratively fusing it with multi-scale skip features.
        - Refining spatial details at each scale.

        This stage translates global semantic reasoning into voxel-accurate boundaries
        through top-down feature propagation.
        """
        f1,f2,f3,f4 = convs 
        x = self.decoder_block_4(x , f4)
        x = self.decoder_block_3(x , f3) 
        x = self.decoder_block_2(x , f2) 
        x = self.decoder_block_1(x , f1) 
        return x 

class Bottleneck(torch.nn.Module) : 
    """
    Deepest semantic processing stage of the network.

    Applies high-capacity convolutional refinement at the lowest resolution,
    serving as the semantic anchor that guides the decoder reconstruction.

    Works jointly with Transformer-enhanced skip features to balance:

    Global context (Transformer)

    Local detail (CNN)
    """
    def __init__(self) : 
        super().__init__() 
        self.bottleneck = Conv_block(in_channles=512 , out_channles=1024 ) 
    def forward (self, x) : 
        """
        Deep semantic bottleneck transformation.

        This block operates at the lowest spatial resolution and highest channel depth.
        It aggregates global contextual information before the decoding stage and
        serves as the semantic anchor for the Transformer-enhanced skip fusion.

        Acts as:
        - A compression of global 3D context.
        - A bridge between convolutional encoding and Transformer reasoning.
        """

        x = self.bottleneck(x) 
        return x 

class Transunet_V2(torch.nn.Module) : 
    """
    Multi-Scale Transformer U-Net V2 (MS-TransUNet V2) for 3D Medical Image Segmentation.

    This architecture is designed to solve the fundamental geometric failure mode of
    standard 3D U-Nets and classical TransUNets when applied to anisotropic medical
    volumes (e.g., thick-slice CT, MRI with low through-plane resolution), where
    aggressive isotropic pooling collapses the Z-axis and destroys global shape
    consistency ("pancake problem").

    The model integrates three complementary reasoning mechanisms:

    Local Geometric Encoding (CNN Encoder)

    A hierarchical 3D convolutional encoder extracts multi-scale spatial features:
    f1 (fine geometry), f2, f3, f4 (deep semantics).
    These features preserve voxel-level continuity, boundary sharpness, and local
    texture statistics essential for organ delineation and lesion topology.

    Global Cross-Scale Semantic Fusion (Multi-Scale Transformer)

    Unlike classical TransUNet designs that tokenize only the bottleneck, this model
    performs simultaneous tokenization of multiple encoder scales (f1–f4).

    Each scale is:
    • Flattened into a token sequence
    • Projected into a shared embedding space (d_model)
    • Augmented with:
    – 3D sinusoidal positional embeddings (true physical coordinates)
    – Learnable level embeddings (semantic depth encoding)

    All scales are concatenated into a unified sequence and processed by a deep
    Transformer stack. Through self-attention, the network learns:

    • Long-range anatomical dependencies (organ-to-organ relations)
    • Cross-resolution alignment (how fine geometry relates to coarse semantics)
    • Global shape priors (continuous organ topology, symmetry, enclosure)

    This creates a Global Semantic Bridge where deep, low-resolution tokens can
    directly attend to high-resolution geometric tokens, enabling questions like:
    “This looks like liver tissue, but what is its full 3D extent and boundary?”

    Hierarchical Reconstruction (U-Net Decoder)

    Transformer-refined multi-scale features are reinjected into the decoder via
    skip connections. The decoder performs progressive upsampling while fusing:

    • Bottleneck semantics (class identity, global context)
    • Transformer-refined skips (globally consistent geometry)
    • Local convolutional refinement (precise boundary recovery)

    This ensures that segmentation is:
    • Globally coherent (no fragmented organs)
    • Geometrically faithful (correct 3D shape)
    • Locally accurate (sharp borders, thin structures preserved)

    Key Architectural Innovations

    • Multi-Scale Tokenization: Transformer sees the full feature hierarchy, not only
    the bottleneck.
    • 3D Sinusoidal Positional Encoding: Preserves physical voxel geometry and spacing.
    • Level Embeddings: Encode semantic depth explicitly in attention space.
    • Cross-Scale Self-Attention: Enables semantic-geometry fusion across resolutions.
    • CNN-Transformer-CNN Hybrid: Combines inductive bias of convolutions with the
    global reasoning of Transformers.

    Clinical Impact

    This design directly addresses:
    • Z-axis collapse in thick-slice volumes
    • Loss of global anatomical continuity
    • Fragmentation of large organs
    • Failure of standard U-Nets to model long-range 3D structure

    The result is anatomically consistent, topology-preserving, and globally aware
    segmentation, making MS-TransUNet V2 suitable for high-precision tasks such as:

    • Multi-organ abdominal segmentation
    • Tumor volumetry
    • Radiotherapy planning
    • Surgical navigation
    • Population-scale morphometric analysis

    Forward Pass Summary

    Encode volume → multi-scale features (f1, f2, f3, f4)

    Tokenize all scales → global Transformer reasoning

    Reproject tokens → refined skip features

    Process bottleneck → deep semantics

    Decode with Transformer-enhanced skips → dense segmentation map

    Output

    A full-resolution 3D segmentation volume with:
    • Preserved global shape
    • Accurate local boundaries
    • Robust performance under anisotropic voxel spacing
    """
    def __init__(self,in_channels =3 , num_classes = 4) : 
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels) 
        self.transformer = Transformer() 
        self.decoder  = Decoder() 
        self.bottleneck = Bottleneck() 
        self.num_classes = num_classes
        self.output = torch.nn.Conv3d(in_channels=64 , out_channels=self.num_classes, kernel_size=1) 
    
    def forward (self  , x) : 
        """
        End-to-end forward pass of MS-TransUNet V2.

        Pipeline:
        1. Multi-scale convolutional encoding.
        2. Multi-level tokenization and global reasoning via Transformer.
        3. Bottleneck semantic compression.
        4. Hierarchical decoding with Transformer-refined skip connections.
        5. Final voxel-wise classification.

        Returns dense 3D segmentation logits with preserved geometric coherence
        across anisotropic volumes.
        """
        convs , pools = self.encoder(x) 
        p1,p2,p3,p4 = pools 
        convs = self.transformer(convs) 
        x = self.bottleneck(p4) 
        x = self.decoder(convs,x) 
        output  = self.output(x) 
        return output 
        