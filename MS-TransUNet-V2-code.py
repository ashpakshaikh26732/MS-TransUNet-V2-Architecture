import torch
import math

class Conv_block(torch.nn.Module) :
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
        for layer in self.conv_layers :
            x = layer(x)

        return x




class Encoder_block (torch.nn.Module) :
    def __init__(self , in_channles , out_channles , pool_size = (2,2,2) , dropout = 0.3 ) :
        super().__init__()
        self.conv_block = Conv_block(in_channles=in_channles , out_channles=out_channles)
        self.pool = torch.nn.MaxPool3d(kernel_size=pool_size )
        self.dropout = torch.nn.Dropout(p = dropout  , inplace = True)

    def forward(self, x) :
        f = self.conv_block(x)
        p = self.pool(f)
        p = self.dropout(p)
        return f , p


class Encoder(torch.nn.Module) :
    def __init__(self , in_channels) :
        super().__init__()
        self.encoder_block_1 = Encoder_block(in_channles=in_channels , out_channles=64)
        self.encoder_block_2 = Encoder_block(in_channles=64 , out_channles=128)
        self.encoder_block_3 = Encoder_block(in_channles=128 , out_channles=256)
        self.encoder_block_4 = Encoder_block(in_channles=256 , out_channles=512)

    def forward (self , x) :
        f1 , p1  = self.encoder_block_1(x)
        f2 , p2 = self.encoder_block_2(p1)
        f3,p3 = self.encoder_block_3(p2)
        f4,p4 = self.encoder_block_4(p3)
        return (f1,f2,f3,f4) ,(p1,p2,p3,p4)



class MLP(torch.nn.Module) :
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
    def __init__(self ,num_heads = 8 , key_dim=64 , d_model=512  , dropout_rate =0.1  ) :
        super().__init__()
        self.multihead_attention = torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads , dropout=dropout_rate, batch_first=True)
        self.mlp = MLP(d_model , d_model)
        self.mlp_dropout = torch.nn.Dropout(p = dropout_rate)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)


    def forward(self, x):
        h = self.norm1(x)
        attn, _ = self.multihead_attention(h, h, h, need_weights=False)
        x = x + attn

        h = self.norm2(x)
        x = x + self.mlp_dropout(self.mlp(h))
        return x


class Transformer(torch.nn.Module) :
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

    def get_3d_sincos_pos_embed(self, d_model , D, H,W , device) :
        assert d_model %  6 ==0
        d = d_model // 3

        def get_1d_pos_embd (dim , length)  :
            pos = torch.arange(length  , device=device).float()
            div  = torch.exp(torch.arange(0,dim,2 , device = device).float() * (-math.log(10000) / dim ))
            pe = torch.zeros(length , dim , device = device)
            pe[:,0::2] = torch.sin (pos[:,None] *div)
            pe[:,1::2] = torch.cos(pos[:,None] * div)

            return pe

        pe_d = get_1d_pos_embd (d ,D)
        pe_h = get_1d_pos_embd (d , H)
        pe_w  = get_1d_pos_embd (d , W)

        pe = (
            pe_d[:, None, None, :].repeat(1, H, W, 1),
            pe_h[None, :, None, :].repeat(D, 1, W, 1),
            pe_w[None, None, :, :].repeat(D, H, 1, 1),
        )

        pe = torch.cat(pe, dim=-1)
        pe = pe.view(-1, d_model)
        return pe

    def enoode_level(self, f : torch.Tensor , proj  , level_id : int )->torch.Tensor :
        b,c,d,h,w = f.shape
        x = f.flatten(2).transpose(1,2)
        x  = proj (x)

        pe_spatial = self.get_3d_sincos_pos_embed(self.d_model , d,h,w,x.device)
        pe_spatial = pe_spatial.unsqueeze(0)

        pe_level = self.level_embedding[level_id].view(1,1,-1)

        return x + pe_spatial + pe_level



    def forward(self , inputs) :
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

        x = self.conv_transpose(x) 
        x = torch.cat([x,f] ,dim = 1 )
        x = self.dropout(x)
        x = self.conv_block(x) 
        return x 

class Decoder (torch.nn.Module) : 
    def __init__(self) : 
        super().__init__()
        self.decoder_block_1 = Decoder_Block(in_channles_conv_trans=128 , out_channels_conv_trans=64 , in_channels_convs=128 ,out_channels_convs=64 )
        self.decoder_block_2 = Decoder_Block(in_channles_conv_trans=256 , out_channels_conv_trans=128 , in_channels_convs=256 ,out_channels_convs=128 )
        self.decoder_block_3 = Decoder_Block(in_channles_conv_trans=512 , out_channels_conv_trans=256 , in_channels_convs=512 ,out_channels_convs=256 )
        self.decoder_block_4 = Decoder_Block(in_channles_conv_trans=1024 , out_channels_conv_trans=512 , in_channels_convs=1024 ,out_channels_convs=512 )
    
    def forward(self , convs  ,x) : 
        f1,f2,f3,f4 = convs 
        x = self.decoder_block_4(x , f4)
        x = self.decoder_block_3(x , f3) 
        x = self.decoder_block_2(x , f2) 
        x = self.decoder_block_1(x , f1) 
        return x 

class Bottleneck(torch.nn.Module) : 
    def __init__(self) : 
        super().__init__() 
        self.bottleneck = Conv_block(in_channles=512 , out_channles=1024 ) 
    def forward (self, x) : 
        x = self.bottleneck(x) 
        return x 

class Transunet_V2(torch.nn.Module) : 
    def __init__(self,in_channels =3 , num_classes = 4) : 
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels) 
        self.transformer = Transformer() 
        self.decoder  = Decoder() 
        self.bottleneck = Bottleneck() 
        self.num_classes = num_classes
        self.output = torch.nn.Conv3d(in_channels=64 , out_channels=self.num_classes, kernel_size=1) 
    
    def forward (self  , x) : 
        convs , pools = self.encoder(x) 
        p1,p2,p3,p4 = pools 
        convs = self.transformer(convs) 
        x = self.bottleneck(p4) 
        x = self.decoder(convs,x) 
        output  = self.output(x) 
        return output 
        