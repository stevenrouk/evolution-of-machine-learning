# 3 Topics - with labels
for i in $(seq 0 2)
do
    echo "3 Topics - #$i"
    python src/analysis/topics_over_time.py topic-evolution --topic-idx $i --n-components 3 --start-year 2000 --end-year 2019 --use-topic-names --output images/topic_evolution_3/topic_"$i"_evolution.png
done

# 3 Topics - with labels - with outliers
for i in $(seq 0 2)
do
    echo "3 Topics (w/ outliers) - #$i"
    python src/analysis/topics_over_time.py topic-evolution --topic-idx $i --n-components 3 --start-year 2000 --end-year 2019 --use-topic-names --show-outliers --output images/topic_evolution_3_with_outliers/topic_"$i"_evolution.png
done

# 10 Topics - with labels
# for i in $(seq 0 9)
# do
#     echo "10 Topics - #$i"
#     python src/analysis/topics_over_time.py topic-evolution --topic-idx $i --n-components 10 --start-year 2000 --end-year 2019 --use-topic-names --output images/topic_evolution_10/topic_"$i"_evolution.png
# done

# 10 Topics - with labels - with outliers
# for i in $(seq 0 9)
# do
#     echo "10 Topics (w/ outliers) - #$i"
#     python src/analysis/topics_over_time.py topic-evolution --topic-idx $i --n-components 10 --start-year 2000 --end-year 2019 --use-topic-names --show-outliers --output images/topic_evolution_10_with_outliers/topic_"$i"_evolution.png
# done

# 20 Topics - without labels
# for i in $(seq 0 19)
# do
#     echo "20 Topics - #$i"
#     python src/analysis/topics_over_time.py topic-evolution --topic-idx $i --n-components 20 --start-year 2000 --end-year 2019 --use-topic-names --output images/topic_evolution_20/topic_"$i"_evolution.png
# done

# 20 Topics - without labels - with outliers
# for i in $(seq 0 19)
# do
#     echo "20 Topics (w/ outliers) - #$i"
#     python src/analysis/topics_over_time.py topic-evolution --topic-idx $i --n-components 20 --start-year 2000 --end-year 2019 --use-topic-names --show-outliers --output images/topic_evolution_20_with_outliers/topic_"$i"_evolution.png
# done