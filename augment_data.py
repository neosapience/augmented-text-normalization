from DataAugmenter import DataAugmenter
import argparse
import threading
from queue import Queue
import os
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--augment_from_scratch", action="store_true", default=False, help="Make the data from scratch. Input path will be ignored.")
    parser.add_argument("--sentence_num_from_scratch", type=int, default=100, help="Number of sentences to generate from scratch. Since each data is word-level, you will have more data than this.")
    parser.add_argument("--sentence_per_generation", type=int, default=1, help="Number of sentences to generate for each API call. Having larger number will reduce the cost, but has a risk of performance degradation.")
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="./augmented")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--suppress_error_reports", action="store_true", help="Suprresses error reports in automatic retry.")
    args = parser.parse_args()
    return args

def parallel_augmentation(idx, args, sentence_list, queue):
    data_points = []
    for sentence in tqdm(sentence_list[idx*(int(len(sentence_list)/args.augment_workers)):(idx+1)*(int(len(sentence_list)/args.augment_workers))]):
        augmenter = DataAugmenter(idx, args, sentence)
        normalized = augmenter.augment()
        if normalized:
            data_points.append(normalized)
    queue.put(data_points)

def parallel_augmentation_scratch(idx, args, queue):
    data_points = []
    for i in range(0, int(args.sentence_num_from_scratch/args.sentence_per_generation/args.workers)):
        augmenter = DataAugmenter(idx, args)
        normalized = augmenter.augment_from_scratch()
        if normalized:
            data_points.extend(normalized)
    queue.put(data_points)

if __name__ == "__main__":
    args = parse_args()
    with open("api_key.txt", "r") as f:
        args.openai_api_key = f.read().strip("\n ")

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    augmented_data = []
    threads = []
    result_queue = Queue()
    if not args.augment_from_scratch and args.input_path:
        with open(args.input_path, "r") as f:
            sentence_list = f.readlines()
        if args.sentence_num_from_scratch % args.workers != 0:
            args.workers = int(args.sentence_num_from_scratch/int((args.sentence_num_from_scratch / args.workers)+1))
        for i in range(0, args.workers):
            thread = threading.Thread(target=parallel_augmentation, args=(i, args, sentence_list, result_queue))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
    
    elif not args.augment_from_scratch and not args.input_path:
        raise ValueError("Set input_path or set augment_from_scratch True.")
    else:
        if int(args.sentence_num_from_scratch/args.sentence_per_generation) % args.workers != 0:
            args.workers = int(int(args.sentence_num_from_scratch/args.sentence_per_generation) / int(int(args.sentence_num_from_scratch/args.sentence_per_generation) / args.workers + 1))
        for i in range(0, args.workers):
            thread = threading.Thread(target=parallel_augmentation_scratch, args=(i, args, result_queue))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
    
    while not result_queue.empty():
        result = result_queue.get()
        augmented_data.extend(result)
    
    google_style_data = ""
    for ad in augmented_data:
        if ad["original_chunk"] == ad["normalized_chunk"]:
            ad["normalized_chunk"] = "<self>"
        google_style_data += ad["semiotic_class"] + "\t" + ad["original_chunk"] + "\t" + ad["normalized_chunk"] + "\n"
    
    with open(os.path.join(args.output_path, "google_style_data.tsv"), "w") as f:
        f.write(google_style_data)
    
    with open(os.path.join(args.output_path, "full_data.json"), "w") as f:
        json.dump(augmented_data, f, indent=4)
