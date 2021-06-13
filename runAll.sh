for i in {0..10}
do
    echo $i
    ./run.sh experiments/experiment_multi_agent/proper_exps.py $i &
done
