import os
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from data import QAData
from unified_data import UnifiedQAData
from t5 import MyT5

def run(args, logger):
    args.size = args.size.lower()
    assert args.size in ["base", "large", "3b"]
    model_name = "allenai/unifiedqa-t5-" + args.size
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if args.is_unifiedqa:
        dev_data = UnifiedQAData(logger, args, args.predict_file, False)
    else:
        dev_data = QAData(logger, args, args.predict_file, False)

    if not args.skip_inference:
        dev_data.load_dataset(tokenizer)
        dev_data.load_dataloader()

    if args.do_train:
        if args.is_unifiedqa:
            train_data = UnifiedQAData(logger, args, args.train_file, True)
        else:
            train_data = QAData(logger, args, args.train_file, True)
        train_data.load_dataset(tokenizer)
        train_data.load_dataloader()

        if args.checkpoint is not None:
            model = MyT5.from_pretrained(model_name,
                                         state_dict=torch.load(args.checkpoint))
        else:
            model = MyT5.from_pretrained(model_name)
        if args.n_gpu>1:
            model = torch.nn.DataParallel(model)
        if args.n_gpu>0:
            model.to(torch.device("cuda"))

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps=args.warmup_steps,
                                        num_training_steps=100000)
        train(args, logger, model, train_data, dev_data, optimizer, scheduler, tokenizer)

    if args.do_predict:
        checkpoint = os.path.join(args.output_dir, 'best-model.pt') if args.checkpoint is None else args.checkpoint
        model = MyT5.from_pretrained(model_name,
                                     state_dict=torch.load(checkpoint))
        logger.info("Loading checkpoint from {}".format(checkpoint))
        if args.n_gpu>0:
            model.to(torch.device("cuda"))
        model.eval()
        ems = inference(logger, model, dev_data, save_predictions=True)
        logger.info("%s on %s data: %.2f" % (dev_data.metric, dev_data.data_type, np.mean(ems)*100))

def train(args, logger, model, train_data, dev_data, optimizer, scheduler, tokenizer):
    model.train()
    global_step = 0
    train_losses = []
    best_accuracy = -1
    stop_training=False

    if args.checkpoint_step > 0:
        for _ in range(args.checkpoint_step):
            global_step += 1
            scheduler.step()

    def convert_to_single_gpu(state_dict):
        def _convert(key):
            if key.startswith('module.'):
                return key[7:]
            return key
        return {_convert(key):value for key, value in state_dict.items()}

    logger.info("Starting training!")
    # dev_data_my = dev_data.dataloader
    for epoch in range(int(args.num_train_epochs)):
        dev_batch_cnt = 0  # add by sqc
        for i, batch in enumerate(tqdm(train_data.dataloader, desc='Training Epoch %i' % epoch, ncols=120)):
            if i%2 == 0:
                for j, batch_dev in enumerate(dev_data.dataloader):
                    this_id = i % len(dev_data.dataloader)
                    if this_id == j:
                        print(i, j)
                        break

            # eval
            if global_step % args.eval_period == 0 and global_step > 0:
                torch.cuda.empty_cache()
                if args.skip_inference:
                    logger.info("Step %d (epoch %d) Train loss %.6f" % (
                            global_step,
                            epoch,
                            np.mean(train_losses)))
                    train_losses = []
                    model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                    if args.n_gpu > 1:
                        model_state_dict = convert_to_single_gpu(model_state_dict)
                    torch.save(model_state_dict, os.path.join(args.output_dir,
                                                              "best-model-{}.pt".format(str(global_step).zfill(6))))
                else:
                    model.eval()
                    # eva loss calc
                    with torch.no_grad():
                        len_batch = len(dev_data_my)
                        if dev_batch_cnt > len_batch:
                            dev_batch_cnt = 0
                        batch_dev = dev_data_my[dev_batch_cnt]
                        dev_batch_cnt += 1
                        batch_dev = [b.to(torch.device("cuda")) for b in batch_dev]
                        labels_dev = batch_dev[2]
                        decoder_input_ids = labels_dev.new_zeros(labels_dev.shape)
                        decoder_input_ids[..., 1:] = labels_dev[..., :-1].clone()
                        decoder_input_ids[..., 0] = model.config.decoder_start_token_id
                        loss_dev = model(input_ids=batch_dev[0], attention_mask=batch_dev[1],
                             decoder_input_ids=decoder_input_ids, decoder_attention_mask=batch_dev[3], labels=labels_dev)[0]
                        train_losses.append(loss.detach().cpu())
                        if global_step % args.log_period == 0:
                            logger.info(
                                "Step %d (epoch %d) Train loss %.6f Accumulated dev loss %.6f " % (
                                    global_step,
                                    epoch,
                                    loss_dev[-1].item(),
                                    np.mean(loss_dev),
                                ))

                    # end: eva loss calc

                    curr_em = inference(logger, model if args.n_gpu==1 else model.module, dev_data)
                    logger.info("Step %d Train loss %.2f %s %.2f%% on epoch=%d" % (
                            global_step,
                            np.mean(train_losses),
                            dev_data.metric,
                            curr_em*100,
                            epoch))
                    train_losses = []
                    if best_accuracy < curr_em:
                        model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                        if args.n_gpu > 1:
                            model_state_dict = convert_to_single_gpu(model_state_dict)
                        torch.save(model_state_dict, os.path.join(args.output_dir, "best-model-%d-%.4f.pt" % (global_step, curr_em)))
                        logger.info("Saving model with best %s: %.2f%% -> %.2f%% on epoch=%d, global_step=%d" % \
                                (dev_data.metric, best_accuracy*100.0, curr_em*100.0, epoch, global_step))
                        best_accuracy = curr_em
                        wait_step = 0
                        stop_training = False
                    else:
                        wait_step += 1
                        logger.info("Waiting %d/%d Best %s: %.2f%%" % \
                                (wait_step, args.wait_step, dev_data.metric, best_accuracy*100.0))
                        if wait_step >= args.wait_step:
                            stop_training = True
                            break
                model.train()
                torch.cuda.empty_cache()
            
            # train
            with torch.enable_grad():
                global_step += 1
                batch = [b.to(torch.device("cuda")) for b in batch]
                labels = batch[2]
                decoder_input_ids = labels.new_zeros(labels.shape)
                decoder_input_ids[..., 1:] = labels[..., :-1].clone()
                decoder_input_ids[..., 0] = model.config.decoder_start_token_id
                loss = model(input_ids=batch[0], attention_mask=batch[1],
                             decoder_input_ids=decoder_input_ids, decoder_attention_mask=batch[3], labels=labels)[0]
                if args.n_gpu > 1:
                    loss = loss.mean()
                if torch.isnan(loss).data:
                    logger.info("Stop training because loss=%s" % (loss.data))
                    stop_training=True
                    break
                train_losses.append(loss.detach().cpu())
                loss.backward()
                if global_step % args.log_period == 0:
                    logger.info("Step %d (epoch %d) Train loss %.6f Accumulated train loss %.6f Learning Rate %f" % (
                                global_step,
                                epoch,
                                train_losses[-1].item(),
                                np.mean(train_losses),
                                optimizer.param_groups[0]['lr']))

                if global_step % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

        if stop_training:
            break


def inference(logger, model, dev_data, save_predictions=True):
    predictions = []
    bos_token_id = dev_data.tokenizer.bos_token_id
    for i, batch in enumerate(tqdm(dev_data.dataloader, desc="Inference", ncols=120)):
        batch = [b.to(torch.device("cuda")) for b in batch]
        outputs = model.generate(input_ids=batch[0],
                                 attention_mask=batch[1],
                                 num_beams=dev_data.args.num_beams,
                                 min_lnegth=1,
                                 max_length=dev_data.args.max_output_length,
                                 early_stopping=True,)
        for input_, output in zip(batch[0], outputs):
            pred = dev_data.decode(output)
            predictions.append(pred)
        if i % dev_data.args.log_period == 0:
            logger.info("Last 5 inference result: {}".format(predictions[-5:])) 
    if save_predictions:
        dev_data.save_predictions(predictions)
    return np.mean(dev_data.evaluate(predictions))







