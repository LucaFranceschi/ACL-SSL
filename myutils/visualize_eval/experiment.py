import load_utils
import viz_utils
import importlib
importlib.reload(load_utils)
importlib.reload(viz_utils)

from load_utils import *
from viz_utils import *

if "EXPERIMENTS" not in globals():
    global EXPERIMENTS
    EXPERIMENTS = {}

def cleanup():
    global EXPERIMENTS
    EXPERIMENTS = {}

def multiton(cls):
    global EXPERIMENTS
    def getinstance(name):
        if name not in EXPERIMENTS:
            EXPERIMENTS[name] = cls(name)
        return EXPERIMENTS[name]
    return getinstance

@multiton
class Experiment(object):
    def __init__(self, exp_name) -> None:
        self.name = exp_name
        self.metrics = pd.DataFrame()
        self.infer_info = pd.DataFrame()
        self.loaded_dirs = []
        self.thresholds = None

    def cleanup(self):
        global EXPERIMENTS
        del EXPERIMENTS[self.name]

    def load_eval_metrics(self, path_to_tensorboard_dir):
        if path_to_tensorboard_dir not in self.loaded_dirs:
            metrics = load_eval(path_to_tensorboard_dir, self.name)
            self.metrics = pd.concat([
                self.metrics,
                metrics
            ])
            self.loaded_dirs.append(path_to_tensorboard_dir)

    def load_eval_inference_info(self, path_to_numpy_dir):
        if path_to_numpy_dir not in self.loaded_dirs:
            self.infer_info = load_infer_info(path_to_numpy_dir, self.name)

            if self.thresholds is None:
                self.thresholds = get_thresholds(path_to_numpy_dir)

            self.loaded_dirs.append(path_to_numpy_dir)

    def _print_metrics(self, epoch, thr, seg_item, snr=False) -> pd.DataFrame | None:
        if self.thresholds and seg_item in self.thresholds[epoch] and thr in self.thresholds[epoch][seg_item]:
            thr = str(self.thresholds[epoch][seg_item][thr])
        df = print_metrics(self.metrics, epoch=epoch, thr=thr, seg_item=seg_item, snr=snr)
        if df is not None:
            df = df.reset_index()
            df['model'] = self.name
            df = df.set_index(['model', 'epoch', 'dataset'])
        return df

    def print_metrics(self, thr, seg_item, snr=False):
        if self.thresholds is None:
            raise Exception('Load inference info first!!')
        df = None
        for i in sorted(self.thresholds.keys()):
            pivot_df = self._print_metrics(i, thr=thr, seg_item=seg_item, snr=snr)
            if pivot_df is not None:
                if df is None:
                    df = pivot_df
                else:
                    df = pd.concat([df, pivot_df])
        if df is not None:
            print(df)
            df.to_clipboard(header=True, sep='\t')
            os.makedirs(f'outputs/{self.name}', exist_ok=True)
            df.to_csv(f'outputs/{self.name}/{"noisy-" if snr else ""}{seg_item}-{thr}.csv')

    def plot_all_metrics(self, epoch, thr, seg_item):
        plot_all_metrics(self.metrics[self.metrics['epoch'] == epoch], thr=thr, seg_item=seg_item, experiment_name=self.name, epoch=epoch)

    def boxplots_by_dataset(self, dataset_name, epochs, th_name, seg_item, min_max='max'):
        boxplots_by_dataset(self.infer_info, dataset_name, self.thresholds, epochs,
                            th_name=th_name, min_max=min_max, seg_item=seg_item, experiment_name=self.name)

    def describe(self, dataset=None, epoch=None, audio_type=None, snr=None, min_max=None, seg_item=None, run=None):
        """
        Describe the data arrays in infer_info filtered by the given criteria.
        Only filters by parameters that are not None. Unspecified columns show all values.
        Results are sorted by audio_type in order: pos, sil, noi, off

        Parameters
        ----------
        dataset : str, optional
        epoch : str/int, optional
        audio_type : str, optional
        snr : int, optional
        min_max : str, optional
        seg_item : str, optional
        run : str, optional

        Returns
        -------
        pd.DataFrame
            Descriptive statistics (count, mean, std, min, q25, q50, q75, max)
        """
        df = self.infer_info.copy()

        # Apply filters for each non-None parameter
        if dataset is not None:
            df = df[df['dataset'] == dataset]
        if epoch is not None:
            df = df[df['epoch'] == epoch]
        if audio_type is not None:
            df = df[df['audio_type'] == audio_type]
        if snr is not None:
            df = df[df['snr'] == snr]
        if min_max is not None:
            df = df[df['min_max'] == min_max]
        if seg_item is not None:
            df = df[df['seg_item'] == seg_item]
        if run is not None:
            df = df[df['run'] == run]

        # Extract descriptive statistics from data arrays
        results = []
        for idx, row in df.iterrows():
            data_array = row['data']
            stats = {
                'epoch': row['epoch'],
                'audio_type': row['audio_type'],
                'snr': row['snr'],
                'dataset': row['dataset'],
                'min_max': row['min_max'],
                'seg_item': row['seg_item'],
                'run': row['run'],
                'count': len(data_array),
                'mean': np.mean(data_array),
                'std': np.std(data_array),
                'min': np.min(data_array),
                'q25': np.percentile(data_array, 25),
                'q50': np.percentile(data_array, 50),
                'q75': np.percentile(data_array, 75),
                'max': np.max(data_array),
            }
            results.append(stats)

        result_df = pd.DataFrame(results)

        # Sort by audio_type in custom order: pos, sil, noi, off
        audio_type_order = pd.CategoricalDtype(categories=['pos', 'sil', 'noi', 'off'], ordered=True)
        result_df['audio_type'] = result_df['audio_type'].astype(audio_type_order)
        result_df = result_df.sort_values('audio_type')

        return result_df


def print_all_metrics(experiments_epochs, thr, seg_item, snr=False):
    '''
    experiments_epochs = {
        'ACL_baseline': ['best'],
        'ACL_v1_B16': [17],
        'ACL_v1_B32': [19],
        'ACL_v2_B16': [16],
        'ACL_v3_B16': [15],
        'ACL_v4_B16': [18],
        'ACL_v5_B16': [x],
    }
    '''
    df = None
    for exp_name, epoch_list in experiments_epochs.items():
        exp = Experiment(exp_name)
        if exp.thresholds is None:
            raise Exception('Load inference info first!!')
        for e in epoch_list:
            # if exp_name == 'frank':
            if exp_name == 'ADCL_v1_B16':
                pivot_df = exp._print_metrics(e, thr=thr, seg_item='v_d', snr=snr)
            else:
                pivot_df = exp._print_metrics(e, thr=thr, seg_item=seg_item, snr=snr)
            if pivot_df is not None:
                if df is None:
                    df = pivot_df
                else:
                    df = pd.concat([df, pivot_df])
    return df