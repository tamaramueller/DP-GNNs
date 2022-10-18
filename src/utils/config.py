import os

PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)

## logging configs
log_path = os.path.join(PROJECT_ROOT, "experiments", "logs")


# default parameter setting for synthetic dataset
nr_classes = 2
connectivity_list = [0.2, 0.3]
means = [0, 0.1]
std_devs = [0.5, 0.5]
