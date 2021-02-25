import os
import argparse

from visualize import Visualizer
from train import Trainer
from utility_functions import print_epoch, update_results_dictionary

def main(args):

    if not os.path.exists(args.models_dir):
        os.mkdir(args.models_dir)
    if not os.path.exists(args.plots_dir):
        os.mkdir(args.plots_dir)

    visualizer = Visualizer(args.fonts_dir, args.plots_dir)
    trainer = Trainer(args)

    if args.resume_training:
        trainer.resume_epoch(args.restore_epoch)

    for epoch in range(trainer.start_epoch, trainer.end_epoch):
        train_dict = trainer.train()
        val_dict = trainer.validate()

        update_results_dictionary(trainer.train_dict, train_dict)
        update_results_dictionary(trainer.val_dict, val_dict)

        trainer.epoch_to_tensorboard(epoch)

        if (epoch + 1) % args.tb_plot_interval == 0:
            lr_ct, sr_ct, hr_ct = trainer.make_ct_trio()
            generated_fig = visualizer.get_generated_ct_fig(lr_ct, sr_ct, hr_ct,
                                                              epoch + 1)
            trainer.tb_writer.add_figure('Epoch {:d}'.format(epoch + 1), generated_fig)

        if (epoch + 1) % trainer.log_interval == 0 or (epoch + 1) == trainer.end_epoch:
            print_epoch(epoch + 1, train_dict, None)

        if (epoch + 1) % trainer.checkpoint_interval == 0:
            trainer.save_checkpoint(epoch + 1)
            lr_ct, sr_ct, hr_ct = trainer.make_ct_trio()
            visualizer.visualize_generated_ct(lr_ct, sr_ct, hr_ct,
                                              epoch + 1)
            visualizer.visualize_performance(trainer.train_dict, trainer.val_dict,
                                             epoch + 1)

        if args.decay_interval and (epoch + 1) % args.decay_interval == 0:
            trainer.update_learning_rate(epoch + 1)

        trainer.tb_writer.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir', type=str, default='.')
    parser.add_argument('--train_dir', type=str, default='./Data/Train')
    parser.add_argument('--val_dir', type=str, default='./Data/Validation')
    parser.add_argument('--test_dir', type=str, default='./Data/Test')
    parser.add_argument('--models_dir', type=str, default='./Models')
    parser.add_argument('--plots_dir', type=str, default='./Plots')
    parser.add_argument('--fonts_dir', type=str, default='./Fonts')

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mode', type=str, default='train')

    # TODO Change checkpoint_interval = 500
    # TODO Change num_epochs = 500
    parser.add_argument('--checkpoint_interval', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--tb_plot_interval', type=int, default=5)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=100)

    parser.add_argument('--resume_training', type=bool, default=False)
    parser.add_argument('--restore_epoch', type=int, default=0)

    parser.add_argument('--alpha_param', type=float, default=.1)
    parser.add_argument('--lambda_coeff', type=int, default=10)
    parser.add_argument('--n_d_iter', type=int, default=5)

    parser.add_argument('--lr_init', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=1e-1)
    parser.add_argument('--decay_interval', type=int, default=0)

    args = parser.parse_args()
    print(args)
    main(args)
