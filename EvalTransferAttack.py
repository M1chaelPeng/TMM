import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models.tokenization_bert import BertTokenizer
import utils
from models.model_retrieval import ALBEF
from attack import *
from torchvision import transforms
from dataset import pair_dataset_attack
from PIL import Image
from tqdm import tqdm
from transformers import BertForMaskedLM

class Model:
    def __init__(self, config, text_encoder_name, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.model = ALBEF(config=config, text_encoder=text_encoder_name, tokenizer=tokenizer)
        self.ref_model = BertForMaskedLM.from_pretrained(text_encoder_name)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint)
        self.model.load_state_dict(state_dict, strict=False)
        print('Checkpoint loaded from %s' % checkpoint_path)

    def to_device(self, device):
        self.model = self.model.to(device)
        self.ref_model = self.ref_model.to(device)

    def inference(self, images, texts_input, use_embeds):
        return self.model.inference(images, texts_input, use_embeds=use_embeds)

class Evaluation:
    def __init__(self, model, ref_model, data_loader, tokenizer, device, config, args):
        self.model = model
        self.ref_model = ref_model
        self.data_loader = data_loader
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.args = args

    def retrieval_eval(self):
        self.model.eval()
        self.ref_model.eval()

        print('Computing features for evaluation adv...')
        start_time = time.time()

        images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        text_attacker = TextAttacker(self.ref_model, self.tokenizer, self.device, cls=self.args.cls)
        from attack.imageAttack import ImageAttacker, NormType
        image_attacker = ImageAttacker(epsilon=self.config['epsilon'], args=self.args, cls=self.args.cls)
        multi_attacker = MultiModalAttacker(self.model, text_attacker, self.tokenizer, cls=self.args.cls, image_attacker=image_attacker)

        print('Prepare memory')
        num_text = len(self.data_loader.dataset.text)
        num_image = len(self.data_loader.dataset.ann)

        image_feats = torch.zeros(num_image, self.config['embed_dim'])
        image_embeds = torch.zeros(num_image, 577, 768)

        text_feats = torch.zeros(num_text, self.config['embed_dim'])
        text_embeds = torch.zeros(num_text, 30, 768)
        text_atts = torch.zeros(num_text, 30).long()
        adv_example = []
        print('Forward')
        for images, texts, texts_ids in tqdm(self.data_loader, ascii=True):
            images = images.to(self.device)
            if self.args.adv != 0:
                images, texts = multi_attacker.run_transfer_attack(images, texts, self.args, num_iters=self.config['num_iters'])

            texts_input = self.tokenizer(texts, padding='max_length', truncation=True, max_length=30,
                                    return_tensors="pt").to(self.device)
            images_ids = [self.data_loader.dataset.txt2img[i.item()] for i in texts_ids]
            with torch.no_grad():
                images = images_normalize(images)
                output = self.model.inference(images, texts_input, use_embeds=False)
                image_feats[images_ids] = output['image_feat'].cpu().detach()
                image_embeds[images_ids] = output['image_embed'].cpu().detach()
                text_feats[texts_ids] = output['text_feat'].cpu().detach()
                text_embeds[texts_ids] = output['text_embed'].cpu().detach()
                text_atts[texts_ids] = texts_input.attention_mask.cpu().detach()

        if self.args.save_json_name:
            with open(self.args.save_json_name,'w',encoding='utf8') as f:
                json.dump(adv_example, f, ensure_ascii=False, indent=2)

        score_matrix_i2t, score_matrix_t2i = self.retrieval_score(image_feats, image_embeds, text_feats,
                                                                text_embeds, text_atts, num_image, num_text)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Evaluation time {}'.format(total_time_str))

        return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()
    
    def retrieval_score(self, image_feats, image_embeds, text_feats, text_embeds, text_atts, num_image, num_text):
        if self.device is None:
            self.device = image_embeds.device

        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Evaluation Direction Similarity With Bert Attack:'

        sims_matrix = image_feats @ text_feats.t()
        score_matrix_i2t = torch.full((num_image, num_text), -100.0).to(self.device)

        for i, sims in enumerate(metric_logger.log_every(sims_matrix, 50, header)):
            topk_sim, topk_idx = sims.topk(k=self.config['k_test'], dim=0)

            encoder_output = image_embeds[i].repeat(self.config['k_test'], 1, 1).to(self.device)
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(self.device)
            output = self.model.text_encoder(encoder_embeds=text_embeds[topk_idx].to(self.device),
                                        attention_mask=text_atts[topk_idx].to(self.device),
                                        encoder_hidden_states=encoder_output,
                                        encoder_attention_mask=encoder_att,
                                        return_dict=True,
                                        mode='fusion'
                                        )
            score = self.model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
            score_matrix_i2t[i, topk_idx] = score

        sims_matrix = sims_matrix.t()
        score_matrix_t2i = torch.full((num_text, num_image), -100.0).to(self.device)

        for i, sims in enumerate(metric_logger.log_every(sims_matrix, 50, header)):
            topk_sim, topk_idx = sims.topk(k=self.config['k_test'], dim=0)
            encoder_output = image_embeds[topk_idx].to(self.device)
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(self.device)
            output = self.model.text_encoder(encoder_embeds=text_embeds[i].repeat(self.config['k_test'], 1, 1).to(self.device),
                                        attention_mask=text_atts[i].repeat(self.config['k_test'], 1).to(self.device),
                                        encoder_hidden_states=encoder_output,
                                        encoder_attention_mask=encoder_att,
                                        return_dict=True,
                                        mode='fusion'
                                        )
            score = self.model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
            score_matrix_t2i[i, topk_idx] = score

        return score_matrix_i2t, score_matrix_t2i

    def itm_eval(self, score_matrix_i2t, score_matrix_t2i, img2txt, txt2img):
        """
        Evaluate Image-Text Matching (ITM) performance.
        
        Args:
            score_matrix_i2t: Image-to-text similarity scores
            score_matrix_t2i: Text-to-image similarity scores
            img2txt: Dictionary mapping image index to list of text indices
            txt2img: Dictionary mapping text index to image index
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Image-to-text retrieval
        i2t_ranks = []
        i2t_recall = {1: 0, 5: 0, 10: 0}
        
        for img_id, txt_ids in img2txt.items():
            scores = score_matrix_i2t[img_id]
            targets = np.array(txt_ids)
            # Get the best rank among all target texts
            sorted_indices = np.argsort(-scores)
            ranks = []
            for target in targets:
                rank = np.where(sorted_indices == target)[0][0] + 1
                ranks.append(rank)
            rank = min(ranks)  # Best rank (lowest number)
            i2t_ranks.append(rank)
            
            for k in [1, 5, 10]:
                if rank <= k:
                    i2t_recall[k] += 1
        
        i2t_recall = {k: v / len(img2txt) * 100 for k, v in i2t_recall.items()}
        i2t_median_rank = np.median(i2t_ranks)
        i2t_mean_rank = np.mean(i2t_ranks)
        
        # Text-to-image retrieval
        t2i_ranks = []
        t2i_recall = {1: 0, 5: 0, 10: 0}
        
        for txt_id, img_id in txt2img.items():
            scores = score_matrix_t2i[txt_id]
            target = img_id
            sorted_indices = np.argsort(-scores)
            rank = np.where(sorted_indices == target)[0][0] + 1
            t2i_ranks.append(rank)
            
            for k in [1, 5, 10]:
                if rank <= k:
                    t2i_recall[k] += 1
        
        t2i_recall = {k: v / len(txt2img) * 100 for k, v in t2i_recall.items()}
        t2i_median_rank = np.median(t2i_ranks)
        t2i_mean_rank = np.mean(t2i_ranks)
        
        return {
            'i2t_recall@1': i2t_recall[1],
            'i2t_recall@5': i2t_recall[5],
            'i2t_recall@10': i2t_recall[10],
            'i2t_median_rank': i2t_median_rank,
            'i2t_mean_rank': i2t_mean_rank,
            't2i_recall@1': t2i_recall[1],
            't2i_recall@5': t2i_recall[5],
            't2i_recall@10': t2i_recall[10],
            't2i_median_rank': t2i_median_rank,
            't2i_mean_rank': t2i_mean_rank,
        }


def main(args, config):
    device = args.gpu[0]

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating dataset")
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
    ])
    test_dataset = pair_dataset_attack(config['test_file'], test_transform, config['image_root'], args)

    test_loader = DataLoader(test_dataset, batch_size=config['batch_size_test'], num_workers=12)

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    model_handler = Model(config, args.text_encoder, tokenizer)
    model_handler.load_checkpoint(args.checkpoint)
    model_handler.to_device(device)

    eval_handler = Evaluation(model_handler.model, model_handler.ref_model, test_loader, tokenizer, device, config, args)
    score_i2t, score_t2i = eval_handler.retrieval_eval()
    result = eval_handler.itm_eval(score_i2t, score_t2i, test_dataset.img2txt, test_dataset.txt2img)
    print(result)
    log_stats = {**{f'test_{k}': v for k, v in result.items()},
                 'eval type': args.adv, 'cls': args.cls, 'eps': config['epsilon'], 'iters':config['num_iters'],'alpha': args.alpha,
                 'intervals': args.intervals, 'num_steps': args.num_steps,
                 'kernel_size': args.kernel_size, 'momentum': args.momentum, 'mode': args.mode, 'epsilon': args.epsilon}
    print(log_stats)
    with open(os.path.join(args.output_dir, args.log_name), "a+") as f:
        f.write(json.dumps(log_stats) + "\n")

    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='flickr')
    parser.add_argument('--config', default='./configs/Retrieval_flickr.yaml')
    parser.add_argument('--output_dir', default='./output/retrieval/flickr')
    parser.add_argument('--checkpoint', default='./checkpoints/ALBEF/flickr30k.pth')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0])
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--adv', default=1, type=int,
                        help='0=clean, 1=adv')
    parser.add_argument('--cls', action='store_true')
    parser.add_argument("--lr", default=3e-4)
    parser.add_argument('--intervals', default=5, type=int,help='number of intervals')
    parser.add_argument('--kernel_size', default=5, type=int,help='kernel size of gaussian filter')
    parser.add_argument('--momentum', default=1, type=float,help='momentum, (default: 1.0)')
    parser.add_argument('--mode', type=str, default="nearest")
    parser.add_argument('--epsilon', default=12, type=float,help='perturbation, (default: 16)')
    parser.add_argument('--epsilon_per', default=0.4, type=float)
    parser.add_argument('--log_name', default="")
    parser.add_argument('--save_json_name', default="")
    parser.add_argument('--config_name', default="")
    parser.add_argument('--save_dir', default="")
    parser.add_argument('--att_mask', default=0.1, type=int)
    parser.add_argument('--alpha', default=0.4, type=float, help='alpha parameter for attack')
    parser.add_argument('--num_steps', default=10, type=int, help='number of attack steps')

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, args.config_name), 'w'))

    main(args, config)
