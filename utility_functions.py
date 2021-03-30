"""
This module contains functions of general utility used in multiple modules.
"""
def update_results_dictionary(current_dict, new_dict):
    """
    Function to update a dictionary according to values in new dictionary.
    :param current_dict: dictionary to update
    :param new_dict: dictionary with data used for updating
    """
    for key, value in new_dict.items():
        if key in current_dict.keys():
            current_dict[key].append(value)


def print_epoch(epoch, train_dict, val_dict=None):
    """
    Function to format the display of epoch log.
    :param epoch: the epoch to print information about
    :param train_dict: dictionary with information about training process
    :param val_dict: dictionary with information about validation process
    """
    print_keys = {'p_loss': 'Perceptual Loss',
                  'psnr': 'PSNR',
                  'sr_range': 'Generated CT values'}

    print('\nEPOCH {:d}'.format(epoch))
    print('-' * 35)

    # Print train information
    print('   Train results')
    for key, message in print_keys.items():
        if key == 'sr_range':
            print('  {:>22s} : {:7.3f} - {:7.3f}'.format(message,
                                                         train_dict[key][0], train_dict[key][1]))
        else:
            print('  {:>22s} : {:7.3f}'.format(message, train_dict[key]))

    # Print validation information
    if val_dict is not None:
        print('\n   Validation results')
        for key, message in print_keys.items():
            if key == 'sr_range':
                print('  {:>22s} : {:7.3f} - {:7.3f}'.format(message,
                                                             val_dict[key][0], val_dict[key][1]))
            else:
                print('  {:>22s} : {:7.3f}'.format(message, val_dict[key]))


def print_params(args_dict):
    """
    Function to format the display of command line arguments dictionary.
    :param args_dict: dictionary of command line arguments
    """
    for key, val in args_dict.items():
        print(f'{key} : {val}')


def fail_format(fail_message):
    """
    Function to format the display of failed operation information.
    :param fail_message: information message to print
    """
    fail_flag = '===FAILED==='

    return '\n{:s}\n   {:s}\n{:s}\n'.format(fail_flag, fail_message, fail_flag)


def success_format(success_message):
    """
    Function to format the display of successful operation information.
    :param success_message: information message to print
    """
    success_flag = '===SUCCEEDED==='

    return '\n{:s}\n   {:s}\n{:s}\n'.format(success_flag, success_message, success_flag)
