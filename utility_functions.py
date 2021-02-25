def update_results_dictionary(current_dict, new_dict):

    for key, value in new_dict.items():
        if key in current_dict.keys():
            current_dict[key].append(value)


def print_epoch(epoch, train_dict, val_dict=None):

    print_keys = {'p_loss': 'Perceptual Loss',
                  'psnr': 'PSNR',
                  'sr_range': 'Generated CT values'}

    print('\nEPOCH {:d}'.format(epoch))
    print('-' * 35)

    print('   Train results')
    for key, message in print_keys.items():
        if key == 'sr_range':
            print('  {:>22s} : {:7.3f} - {:7.3f}'.format(message,
                                                         train_dict[key][0], train_dict[key][1]))
        else:
            print('  {:>22s} : {:7.3f}'.format(message, train_dict[key]))

    if val_dict is not None:
        print('\n   Validation results')
        for key, message in print_keys.items():
            if key == 'sr_range':
                print('  {:>22s} : {:7.3f} - {:7.3f}'.format(message,
                                                             val_dict[key][0], val_dict[key][1]))
            else:
                print('  {:>22s} : {:7.3f}'.format(message, val_dict[key]))


def fail_format(fail_message):
    fail_flag = '===FAILED==='

    return '\n{:s}\n   {:s}\n{:s}\n'.format(fail_flag, fail_message, fail_flag)


def success_format(success_message):
    success_flag = '===SUCCEEDED==='

    return '\n{:s}\n   {:s}\n{:s}\n'.format(success_flag, success_message, success_flag)
