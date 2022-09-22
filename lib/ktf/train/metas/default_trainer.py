class DefaultTrainer:
    """Default Trainer for the "meta" key in a train config.
    This gets used by the Environment object automatically if no "meta" key is specified.
    """

    def __init__(self):
        pass

    def __call__(self,
                 train_dataset,
                 valid_dataset,
                 create_model_fn,
                 create_loss_fn,
                 create_loss_weights_fn,
                 create_metrics_fn,
                 create_optimizer_fn,
                 create_train_loop_fn):
        """Functor to start training sequence

        Args:
            train_dataset: A tf.data.Dataset to use in training
            valid_dataset: A tf.data.Dataset to use in validation
            create_model_fn: A function to create the model for training
            create_loss_fn: A function to create the loss model for training
            create_loss_weights_fn: A function to create the loss weights used for training
            create_metrics_fn: A function to create the metrics for scoring during training
            create_optimizer_fn: A function to create the optimizer used during training
            create_train_loop_fn: A function to create instance of `ktf.train.TrainLoop` class.

        Returns:
            The results of the ktf.train.TrainLoop
        """
        train_loop = create_train_loop_fn()
        return train_loop(train_dataset,
                          valid_dataset,
                          create_model_fn,
                          create_loss_fn,
                          create_loss_weights_fn(),
                          create_metrics_fn,
                          create_optimizer_fn)
