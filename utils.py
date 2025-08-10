import os
import numpy as np
import paddle


def load_pretrained_model(model, pretrained_model):
    if pretrained_model is not None:
        print('Loading pretrained model from {}'.format(pretrained_model))

        if os.path.exists(pretrained_model):
            para_state_dict = paddle.load(pretrained_model)
            model_state_dict = model.state_dict()
            keys = model_state_dict.keys()
            num_params_loaded = 0
            for k in keys:
                if k not in para_state_dict:
                    print('{} is not in pretrained model'.format(k))
                elif list(para_state_dict[k].shape) != list(
                        model_state_dict[k].shape):
                    print("[SKIP] shape of pretrained params {} doesn't match.(Pretrained: {}, Actual:{})"
                          .format(k, para_state_dict[k].shape,
                                  model_state_dict[k].shape))
                else:
                    model_state_dict[k] = para_state_dict[k]
                    num_params_loaded += 1
            model.set_dict(model_state_dict)
            print("There are {}/{} variables loaded into {}."
                  .format(num_params_loaded, len(model_state_dict),
                          model.__class__.__name__))
        else:
            raise ValueError(
                "The pretrained model directory is not Found: {}"
                    .format(pretrained_model)
            )
    else:
        print('No pretrained model to load, {} will be trained from scratch.'
              .format(model.__class__.__name__))


def pd_tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    img = tensor.squeeze().cpu().numpy()
    img = img.clip(min_max[0], min_max[1])
    img = (img - min_max[0]) / (min_max[1] - min_max[0])
    if out_type == np.uint8:
        # scaling
        img = img * 255.0
    img = np.transpose(img, (1, 2, 0))
    img = img.round()
    img = img[:, :, ::-1]
    return img.astype(out_type)