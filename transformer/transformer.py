from .modules import *
from .encoder import Encoder
from .decoder import Decoder
class Transfomer(nn.Module):
    def __init__(self,encoder:Encoder,decoder:Decoder,src_embed:InputEmbeddings,trgt_embed:InputEmbeddings,src_pos:PositionalEncoding,trgt_pos:PositionalEncoding,projection_layer:ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trgt_embed = trgt_embed
        self.src_pos = src_pos
        self.trgt_pos = trgt_pos
        self.projection_layer = projection_layer
    
    def encode(self,src,src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src,src_mask)
    def decode(self,encoder_out,src_mask,trgt,trgt_mask):
        trgt = self.trgt_embed(trgt)
        trgt = self.trgt_pos(trgt)
        return self.decoder(trgt,encoder_out,src_mask,trgt_mask)
    def project(self,x):
        return self.projection_layer(x)
        

