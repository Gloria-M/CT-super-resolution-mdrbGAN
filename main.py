import os
import time
import argparse
from multiprocessing import Process

from visualize import Visualizer
from train import Trainer
from test import Tester
from utility_functions import print_epoch, update_results_dictionary, print_params


def run_tensorboard(args):
    """
    Function to start Tensorboard session.
    :param args: command line arguments
    """
    print('\n\n... Starting Tensorboard ...\n\n')

    if args.tb_port is None:
        os.system("tensorboard --logdir={:s}".format(args.tb_logdir))
    else:
        os.system("tensorboard --logdir={:s} --port={:d}".format(args.tb_logdir, args.tb_port))


def run_train(args):
    """
    Function to train the Generative Adversarial Network based on multiple dense residual blocks.
    :param args: command line arguments
    """
    # Wait for Tensorboard session to start
    time.sleep(30)
    print('\n\n... Starting Training ...\n\n')

    # Create directories for models and plots if they do not exist
    if not os.path.exists(args.models_dir):
        os.mkdir(args.models_dir)
    if not os.path.exists(args.plots_dir):
        os.mkdir(args.plots_dir)

    # Create objects for training and visualization
    visualizer = Visualizer(args.fonts_dir, args.plots_dir)
    trainer = Trainer(args)

    # Resume training from `restore_epoch` if specified
    if args.resume_training:
        trainer.resume_epoch(args.restore_epoch)
        # Write past results to Tensorboard
        trainer.epoch_to_tensorboard(args.restore_epoch, resume=True)

    for epoch in range(trainer.start_epoch, trainer.end_epoch):
        # Train and validate the model
        train_dict = trainer.train()
        val_dict = trainer.validate()

        # Update the train and validation information dictionaries with results for current epoch
        update_results_dictionary(trainer.train_dict, train_dict)
        update_results_dictionary(trainer.val_dict, val_dict)

        # Log current epoch results to Tensorboard
        trainer.epoch_to_tensorboard(epoch, resume=False)

        # Plot CT images triplet to Tensorboard every `tb_plot_interval`
        if (epoch + 1) % args.tb_plot_interval == 0:
            lr_ct, sr_ct, hr_ct = trainer.make_ct_trio()
            generated_fig = visualizer.get_generated_ct_fig(lr_ct, sr_ct, hr_ct,)
            trainer.tb_writer.add_figure('Epoch {:d}'.format(epoch + 1), generated_fig)

        # Display epoch every `log_interval`
        if (epoch + 1) % trainer.log_interval == 0 or (epoch + 1) == trainer.end_epoch:
            print_epoch(epoch + 1, train_dict, val_dict)

        # Save trained models weights and optimizers parameters an visualize CT results
        # every `checkpoint_interval`
        if (epoch + 1) % trainer.checkpoint_interval == 0:
            trainer.save_checkpoint(epoch + 1)
            lr_ct, sr_ct, hr_ct = trainer.make_ct_trio()
            visualizer.visualize_generated_ct(lr_ct, sr_ct, hr_ct, epoch)
            visualizer.visualize_performance(trainer.train_dict, trainer.val_dict, epoch)

        # Update the learing rate every `decay_interval`
        if args.decay_interval and epoch % args.decay_interval == 0:
            trainer.update_learning_rate(epoch)

        trainer.tb_writer.close()

    print('\n\n... Finished Training ...\n\n')


def run_test(args):
    """
    Function to test the Generator.
    :param args: command line arguments
    """
    print('\n\n... Starting Testing ...\n\n')

    # Create directory for plots if it doesn't exist
    if not os.path.exists(args.plots_dir):
        os.mkdir(args.plots_dir)

    # Create objects for testing and visualization
    visualizer = Visualizer(args.fonts_dir, args.plots_dir)
    tester = Tester(args)

    # Load the Generator saved at epoch `restore_epoch`
    tester.load_checkpoint(args.restore_epoch)

    # Print generated CT scores
    for ct_name in args.test_ct_names:
        print('   --- CT {:s} :'.format(ct_name), end='')

        lr_slice, sr_slice, hr_slice, scores_dict = tester.test(ct_name)
        visualizer.visualize_full_slices(lr_slice, sr_slice, hr_slice, scores_dict, ct_name, args.restore_epoch)
        print('{:.3f}'.format(scores_dict['psnr']))

    print('\n\n... Finished Testing ...\n\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dir', type=str, default='Data/Train', help="path to train data directory")
    parser.add_argument('--val_dir', type=str, default='Data/Validation', help="path to validation data directory")
    parser.add_argument('--test_dir', type=str, default='Data/Test', help="path to test data directory")

    parser.add_argument('--fonts_dir', type=str, default='Fonts', help="path to font directory")
    parser.add_argument('--models_dir', type=str, default='Models', help="path to models directory")
    parser.add_argument('--plots_dir', type=str, default='Plots', help="path to plots directory")

    parser.add_argument('--tb_logdir', type=str, default='runs', help="path to Tensorboard logs directory")
    parser.add_argument('--tb_port', type=int, help="port number for Tensorboard session")
    parser.add_argument('--tb_plot_interval', type=int, default=10, help="frequency of plotting to Tensorboard")

    parser.add_argument('--device', type=str, default='cuda', help="use CUDA if available")
    parser.add_argument('--mode', type=str, default='train', help="train | test  - training / testing mode")

    parser.add_argument('--checkpoint_interval', type=int, default=100, help="frequency of models saving")
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=500)

    parser.add_argument('--resume_training', type=bool, default=False, help="specify whether to resume training")
    parser.add_argument('--restore_epoch', type=int, default=0, help="epoch to resume training from")

    parser.add_argument('--alpha_param', type=float, default=.1)
    parser.add_argument('--lambda_coeff', type=int, default=10)
    parser.add_argument('--n_d_iter', type=int, default=5)

    parser.add_argument('--lr_init', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=1e-1)
    parser.add_argument('--decay_interval', type=int, default=0, help="frequency of learning rate decrease")

    parser.add_argument('--test_ct_names', nargs='+', help="names of CT slices used for testing "
                                                           "(without the .npy extension)")

    args = parser.parse_args()

    print('\n\n')
    print_params(args.__dict__)
    print('\n\n')

    if args.mode == 'train':

        process1 = Process(target=run_tensorboard, args=(args,))
        process2 = Process(target=run_train, args=(args,))

        process1.start()
        process2.start()

    elif args.mode == 'test':
        run_test(args)
