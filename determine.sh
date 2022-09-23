for i in {0..4}
do
python c51.py --randstart --iql --determine --path determine --episode 10000 --modelname iqlrd --Lr 0.1 --samplenum 20
python c51.py --randstart --method4 --determine --path determine --episode 10000 --modelname method4rd --Lr 0.1 --samplenum 20
python c51.py --method4 --determine --path determine --episode 10000 --modelname method4 --Lr 0.1 --samplenum 20
python c51.py --iql --determine --path determine --episode 10000 --modelname iql --Lr 0.1 --samplenum 20
done
