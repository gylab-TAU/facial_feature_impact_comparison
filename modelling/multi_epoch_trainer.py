import torch

class MultiEpochTrainer(object):
    def __init__(self, model: torch.nn.modules.Module,
                 optimizer: torch.optim.Optimizer,
                 model_store,
                 train_phase,
                 val_phase,
                 test_phase,
                 lr_scheduler,
                 performance_threshold: float = 1,
                 best_acc1: int = 0,
                 num_epochs_to_test: int = -1):
        assert train_phase is not None
        assert val_phase is not None

        self.model = model
        self.__optimizer = optimizer
        self.__model_store = model_store
        self.__train_phase = train_phase
        self.__val_phase = val_phase
        self.__test_phase = test_phase
        self.__performance_threshold = performance_threshold
        self.__num_epochs_to_test = num_epochs_to_test
        self.__best_acc1 = best_acc1
        self.__lr_scheduler = lr_scheduler

    def train(self, start_epoch, end_epoch):
        for epoch in range(start_epoch, end_epoch):
            train_output = self.__train_phase.run(epoch)
            print(train_output)
            val_output = self.__val_phase.run(epoch)
            print(val_output)

            acc = self.__get_output_acc(val_output)

            if type(self.__lr_scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                self.__lr_scheduler.step(acc)
            else:
                self.__lr_scheduler.step()

            # remember best acc@1 and save checkpoint
            is_best = acc > self.__best_acc1
            self.__best_acc1 = max(acc, self.__best_acc1)

            self.__model_store.save_model(self.model, self.__optimizer, epoch, self.__best_acc1, is_best)

            if self.__should_test(epoch):
                test_output = self.__test_phase.run(epoch)
                acc = self.__get_output_acc(test_output)
                if acc > self.__performance_threshold:
                    print(f'Done in {epoch} epochs')
                    print("Test output: ", test_output)
                    return test_output

    def __get_output_acc(self, output):
        if type(output) is tuple or type(output) is list:
            return output[0]

    def __should_test(self, epoch):
        # If we have a way to test our performance
        # And if we have a valid "epoch step" after which we wish to test our performance
        # And if we the epoch is a natural number of 'epoch steps'
        # Dayeynu
        return self.__test_phase is not None \
               and self.__num_epochs_to_test is not None \
               and epoch % self.__num_epochs_to_test == 0
