rm drop_dataset/cached*
sh eval.sh drop_dataset/drop_dataset_dev.json prediction.json
#sh eval.sh ../data/final/predicted_dev.json prediction.json
python drop_eval.py --gold_path drop_dataset/drop_dataset_dev.json --prediction_path prediction.json
#python drop_eval.py --gold_path ../data/final/predicted_test.json  --prediction_path prediction.json

#python drop_eval.py --gold_path ../data/final/splits/4_split.json --prediction_path prediction.json
#python drop_eval.py --gold_path ../data/no_answer_splits/span_split.json --prediction_path prediction.json
#python drop_eval.py --gold_path ../data/no_answer_splits/binary_split.json --prediction_path prediction.json
#python drop_eval.py --gold_path ../data/no_answer_splits/none_split.json --prediction_path prediction.json
#python drop_eval.py --gold_path ../data/no_answer_splits/value_split.json --prediction_path prediction.json

#python drop_eval.py --gold_path ../data/no_answer_splits/b_split_answers.json --prediction_path prediction.json
#python drop_eval.py --gold_path ../data/no_answer_splits/t_split_answers.json --prediction_path prediction.json
#python drop_eval.py --gold_path ../data/no_answer_splits/m_split_answers.json --prediction_path prediction.json
#python drop_eval.py --gold_path ../data/no_answer_splits/c_split_answers.json --prediction_path prediction.json
