from transformers import AutoTokenizer, T5ForConditionalGeneration

model_name = 'allenai/unifiedqa-t5-large'    # ../3rd_models/unifiedqa-t5-large
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


def run_predict(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    return tokenizer.batch_decode(res, skip_special_tokens=True)


if __name__ == '__main__':
    import time
    begin = time.time()
    task = "Who is jack ?" \
           "\n jack is a boy- cartilage - cartilage cartilage is a resilient and smooth elastic tissue," \
           " a rubber - like padding that covers and protects the ends of long bones at the joints ," \
           " and is a structural component of the rib cage , the ear , the nose , the bronchial tubes , the intervertebral discs , and many other body components . " \
           "it is not as hard and rigid as bone , but it is much stiffer and much less flexible than muscle . jack is eating    " \
           "the matrix of cartilage is made up of chondrin . because of its rigidity , cartilage often serves the purpose of holding tubes open in the body . " \
           "examples include the rings of the trachea , such - - cartilage - compared to other connective tissues , cartilage has a very slow turnover of its extracellular matrix " \
           "and does not repair . in embryogenesis , the skeletal system is derived from the mesoderm germ layer . Jack is a girl"
    res = run_predict(task)
    print(res)
    mid = time.time()
