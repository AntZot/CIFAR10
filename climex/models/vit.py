from typing import Tuple
from typing import Any
from torchmetrics.functional import accuracy, auroc
import torch
from torch import nn
from torch.nn import functional as F 

import lightning as L
from einops import rearrange, repeat

class ImagePathces(nn.Module):
    def __init__(self, img_size, in_ch:int = 3,patch_size: int = 16):
        super().__init__()
        assert img_size % patch_size == 0

        emb_dim = patch_size * patch_size * in_ch
        self.cls_token = nn.Parameter(torch.randn(1,1,emb_dim))

        self.pos_emb = nn.Parameter(torch.randn(1,1+(img_size//patch_size)**2,emb_dim))
        
        self.patches = nn.Conv2d(in_ch,emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.patches(x)

        # x = torch.flatten(x,2)
        # b, e, hw = x.shape
        # x = x.reshape((b,hw,e))
        x = rearrange(x, "b e h w -> b (h w) e")
        
        b, _, _ = x.shape

        #cls_token = self.cls_token.repeat((b,1,1))
        cls_token = repeat(self.cls_token,"() s e -> b s e",b=b)

        x = torch.concat((cls_token, x),dim=1)
        return x + self.pos_emb 


class MLP(nn.Module):
    def __init__(self,input_size: int, hidden_size: int = None, output_size: int = None, droput = 0.,layer_act: torch.nn = nn.GELU) -> None:
        super().__init__()
        if hidden_size is None:
            hidden_size = input_size
        if output_size is None:
            output_size = input_size

        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,output_size)
        self.actv = layer_act()
        self.drop = nn.Dropout(droput)

    def forward(self,x):
        return self.drop(self.fc2(self.drop(self.actv(self.fc1(x)))))


class Attention(nn.Module):
    def __init__(self,dim: int,qkv_bias: bool = False, num_heads: int =8, attn_drop=0., out_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim,dim*3,bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_lin = nn.Linear(dim,dim)
        self.out_drop = nn.Dropout(out_drop)

    def forward(self,x):
        x = self.qkv(x)
        b, e, _ = x.shape
        # b e (q k v)
        x = x.reshape((b,3,self.num_heads,e,-1))
        # b qkv heads e s 
        
        att = F.softmax((x[:,0] @ x[:,1].transpose(2,3))*self.scale, dim=-1)

        att = self.attn_drop(att)
        att = att @ x[:,2]
        #out = out.reshape(b,e,-1)
        att = rearrange(att, "b h n e -> b n (h e)")

        out = self.out_drop(self.out_lin(att))

        return out
    

class Block(nn.Module):
    def __init__(self, dim, qkv_bias: bool = False, num_heads:int = 8, mlp_ratio:int = 4, drop_rate: float = 0.) -> None:
        super().__init__()
        self.att = Attention(dim,num_heads=num_heads,qkv_bias=qkv_bias,attn_drop=drop_rate,out_drop=drop_rate)

        self.att_norm = nn.LayerNorm(dim)

        self.drop = nn.Dropout(drop_rate)

        self.mlp = MLP(dim,dim*mlp_ratio,dim)
        
        self.mlp_norm = nn.LayerNorm(dim)
        
        

    def forward(self, x):
        att = self.att(x)
        att_out = self.att_norm(x + att)

        mlp = self.drop(self.mlp(att_out))
        mlp_out = self.mlp_norm(x + mlp)
        return mlp_out


class Encoder(nn.Module):
    def __init__(self, dim, qkv_bias: bool = False, N:int = 6, num_heads: int = 8, mlp_ratio: int = 4, drop_rate: float = 0.) -> None:
        super().__init__()
        self.blocks =nn.ModuleList([
            Block(dim,qkv_bias,num_heads,mlp_ratio,drop_rate)
        for _ in range(N)
        ])

    def forward(self,x):
        for block in self.blocks:
            x = block(x)
        return x
    

BATCH_SIZE = 64 if torch.cuda.is_available() else 16

class ViT(L.LightningModule):
    def __init__(self, img_size: int, patch_size: int, num_classes: int, 
                 in_ch: int = 3, depth: int=6, num_heads: int = 8, 
                 mlp_ratio: int = 4, qkv_bias=False, drop_rate: float = 0.,
                 lr=1e-5):
        super().__init__()
        
        self.save_hyperparameters()
        self.learning_rate = lr
        self.num_classes = num_classes
        emb_dim = in_ch*patch_size**2
        
        self.patch = ImagePathces(img_size = img_size,
                                  in_ch = in_ch,
                                  patch_size = patch_size)

        self.encoder = Encoder(dim = emb_dim,
                               qkv_bias = qkv_bias,
                               N = depth,
                               num_heads = num_heads,
                               mlp_ratio = mlp_ratio,
                               drop_rate = drop_rate)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes)
        )
        #self.mlp_head = nn.Linear(emb_dim,num_classes)

    def _set_batch_and_dt_size(self,batch_size,size):
        self.batch_size = batch_size
        self.size = size
    
    def forward(self,x):
        x = self.patch(x)

        x = self.encoder(x)

        x = self.mlp_head(x[:,0])
        return x
    

    def training_step(self, batch, batch_nb: int):
        x,y = batch

        preds = self(x)
        loss = F.cross_entropy(preds,y)
        """
        metrics
        """
        acc = accuracy(preds, y, task="multiclass",num_classes=self.num_classes)
        self.log("train_accuracy", acc, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        
        return {'loss': loss, 'prediction': preds}


    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        """
        metrics
        """
        rocauc = auroc(logits,y,task="multiclass",num_classes=self.num_classes)
        acc = accuracy(logits, y, task="multiclass",num_classes=self.num_classes)
        self.log("val_accuracy", acc, prog_bar=True)
        self.log("val_rocauc",rocauc,prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self,batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        x, y = batch
        logits = self(x)
        
        test_loss = F.cross_entropy(logits, y)
        """
        metrics
        """
        rocauc = auroc(logits,y,task="multiclass",num_classes=self.num_classes)
        acc = accuracy(logits, y, task="multiclass",num_classes=self.num_classes)
        self.log("test_accuracy", acc, prog_bar=True)
        self.log("test_rocauc",rocauc,prog_bar=True)
        self.log("test_loss", test_loss, prog_bar=True)

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.learning_rate)

        # scheduler = {
        #     "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
        #         optimizer=optimizer,patience=5),
        #     "monitor": "val_loss"
        # }

        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=2e-5,
                epochs=self.trainer.max_epochs,
                steps_per_epoch = self.trainer.train_dataloader),
            "interval": "step"
        }
        return {"optimizer": optimizer, "lr_scheduler":scheduler}