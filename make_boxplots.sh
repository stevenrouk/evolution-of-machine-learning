for i in $(seq 0 9);
do python src/analysis/topics_over_time.py topic-evolution --topic-idx $i --n-components 10 --start-year 2000 --end-year 2019 --use-topic-names --output images/topic_"$i"_evolution.png;
done