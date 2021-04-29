from data_prep.multi_stage_processing import MultiStageProcessor
import data_prep


class GenericFactory(object):
    def get_processors(self, processors_list: list):
        processors = []
        for proc_config in processors_list:
            params = proc_config['params']
            print(data_prep.__dict__)
            data_prep.__dict__[proc_config['type']](**params)
        return data_prep.MultiStageProcessor(processors)

