import open_clip

model, _, preprocess = open_clip.create_model_and_transforms(
            model_name='ViT-L-14', device='cuda')#pretrained='laion2b_s34b_b79k')

for name,parameter in model.named_parameters():
    print(name)