# Ziheng Huang

**Third year computer engineering major**

# Interests:
1. Deep Learning
2. Computer vision
3. HCI

# Languages :
-   Python
-   Javascript
-   java
-   C++

Quote:
> Start early Start often

# Ongoing projects:

- [ ] Defect Detection
- [ ] 3D Medical Data classification


# Some Current Experiments

![Image of scrl](https://raw.githubusercontent.com/kakaobrain/scrl/master/images/overview.png)


Quote code 

original CVPR paper: [Spacial Consistent Representation Learning](https://arxiv.org/abs/2103.06122):

```
class SCRLModel(nn.Module):
    def __init__(self,):
        super(SCRLModel, self).__init__()
        self.backbone = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).backbone
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.projhead = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, 128))
        self.roi_align = ops.RoIAlign(output_size=(3,3),
                                          sampling_ratio=-1,
                                          spatial_scale=1.0,
                                          aligned=False)

    def forward(self, x,roi):
        x = self.backbone.body(x)['3'] # numSample x 2048 x 8 x 8
        x = self.roi_align(x,roi)
        x = self.avgpool(x) # numSample x 2048 x 1 x 1
        x = torch.flatten(x, 1) # numSample x 2048
        x = self.projhead(x) # numSample x 128
        return x
```


### [link to another file](index2.md)
