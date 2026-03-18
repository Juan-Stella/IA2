import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

save_name = sys.argv[1]
script_to_run = sys.argv[2]

origin_show = plt.show

def my_show(*args, **kwargs):
    plt.savefig(save_name, bbox_inches='tight', dpi=150)
    plt.close()

plt.show = my_show

try:
    with open(script_to_run, "r", encoding="utf-8") as f:
        code = f.read()
    exec(code, {"__name__": "__main__"})
except Exception as e:
    print(f"Error running {script_to_run}: {e}", file=sys.stderr)
