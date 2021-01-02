import data_prep.factories.generic_factory

if __name__ == '__main__':
    existing_ds_root = '/home/administrator/datasets/vggface2/train'
    proc_factory = data_prep.factories.generic_factory.GenericFactory()
    proc_dest_root = '/home/administrator/datasets/processed'
    config = [{'type': 'ClassSizeProcessing', 'params': {'dest_root_dir': proc_dest_root, 'min_size': 350}},
            {'type': 'DlibAlignProcessor', 'params': {'output_dataset_dir': proc_dest_root, }},
            {'type': 'PhaseSizeProcessor', 'params': {'output_dataset_dir': proc_dest_root, 'phase_size_dict': {'train': 300, 'val': 50}}}]
    proc = data_prep.multi_stage_processing.MultiStageProcessor(
        [
            data_prep.class_size_processing.ClassSizeProcessing(**{'output_dataset_dir':proc_dest_root, 'min_size':400}),
            # data_prep.list_exclusion_processing.ListExclusionProcessor(**{'output_dataset_dir': proc_dest_root,'exclusion_list': preexisting_classes}),
            data_prep.num_classes_processing.NumClassProcessor(min_class=1250, max_class=1250,
                                                                output_dataset_dir=proc_dest_root),
            # data_prep.dlib_processor.DlibAlignProcessor(**{'output_dataset_dir': proc_dest_root}),
            data_prep.phase_size_processing.PhaseSizeProcessor(**{'output_dataset_dir': proc_dest_root, 'phase_size_dict': {'train': 300, 'val': 50, 'test':50}})
         ])

    dest, num_cl = proc.process_dataset(existing_ds_root, 'vggface2_discriminator')

    print('Moved ', existing_ds_root, ' to ', dest)
    print('Done.')
