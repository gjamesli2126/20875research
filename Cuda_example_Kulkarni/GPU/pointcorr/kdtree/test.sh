#! /bin/sh

for dir in *
do
		if [[ -d $dir ]] && [[ -f ./$dir/run_test.sh ]]
		then				
				cd $dir
				echo "Executing $dir..."
				source run_test.sh
				cd ..
		fi
done

#exit 0
