import os
import uuid
import datetime

REPEAT=3
ROOT_DEPTH=[16]
BLOCK_DIM=[64,128,256,512,1024]
MVCS = {
	"cop_hat300-1.txt":292,
	"cop_hat300-2.txt":275,
	"cop_hat300-3.txt":264,
	"cop_hat500-1.txt":491,
	"cop_hat500-2.txt":464,
	"cop_hat500-3.txt":450,
	"cop_hat700-1.txt":689,
	"cop_hat700-2.txt":656,
	"cop_hat1000-1.txt":990,
	"cop_hat1000-2.txt":954,
	"CAIDA.txt":3683,
	"Sister-Cities.txt":5527,
	"lastfm-asia.txt":3447,
	"Power-Grid.txt":2203,
	"vc-exact_009.txt":21348,
	"vc-exact_023.txt":16013,
	"formated_wikipedia_link_csb.txt":1998,
	"formated_wikipedia_link_lo.txt":1536,
	"formated_movielens-100k_rating.txt":849,
	"formated_maayan-vidal.txt":1077,
	"formated_maayan-figeys.txt":316,
}

QUEUE_SIZES = {
	"Sister-Cities.txt":131072,
	"lastfm-asia.txt":262144,
	"Power-Grid.txt":524288,
}

VERSION = ["stackonly"]
INSTANCE = ["mvc","pvc"]
K = [-1]

GRAPHS_FOLDER_PATH = "test_graphs"

RUN_GRAPHS = [
	"cop_hat700-2.txt",
	"cop_hat500-3.txt",
	"vc-exact_009.txt",
]


SKIP_GRAPHS = [
]

os.system(f"make --always-make")

for filename in os.listdir(GRAPHS_FOLDER_PATH):
	if (RUN_GRAPHS and (filename in RUN_GRAPHS)) or (not RUN_GRAPHS and not SKIP_GRAPHS) or (SKIP_GRAPHS and (filename not in SKIP_GRAPHS)):
		f = os.path.join(GRAPHS_FOLDER_PATH, filename)

		for v in VERSION:
			for i in INSTANCE:
				exit_code = 1
				for d in ROOT_DEPTH:
					for b in BLOCK_DIM:
						extra_flags_runs = []
						if i == "pvc":
							for k in K:
								extra_flags_runs.append(f" -k {k + MVCS[filename]}")
						else:
							extra_flags_runs.append("")

						for r in extra_flags_runs:
							uid = uuid.uuid4()
							for _ in range(REPEAT):
								ct = datetime.datetime.now()
								os.system(f'echo "\n\nTime : {ct} -----> Graph: {filename} Version : {v}, Instance : {i}, Depth : {d}, BlockDim : {b}, Flag : {r}, Run : {_} \n"')
								command = f"srun --partition=cmps-ai --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --time=2:00:00 --nodelist=onode11 ./output -f {f} -o {uid} -v {v} -i {i} -d {d} -b {b}" + r
								
								exit_code = os.system(command)

								if exit_code != 0:
									break
