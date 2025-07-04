from torch import no_grad, stack
from torch.utils.data import DataLoader
from torch.nn import Module


"""
Functions you should use.
Please avoid importing any other torch functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
from torch.nn import Parameter, Linear
from torch import optim, tensor, tensordot, empty, ones
from torch.nn.functional import cross_entropy, relu, mse_loss
from torch import movedim


class PerceptronModel(Module):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.

        In order for our autograder to detect your weight, initialize it as a
        pytorch Parameter object as follows:

        Parameter(weight_vector)

        where weight_vector is a pytorch Tensor of dimension 'dimensions'


        Hint: You can use ones(dim) to create a tensor of dimension dim.

        感知机用于将数据点分类为属于某个特定类别(+1)或不属于该类别(-1)。`dimensions` 参数表示数据的维度。例如，如果 `dimensions=2`，则表示感知机需要对二维点进行分类。

        为了让我们的自动评分系统能够检测到你的权重，你需要将其初始化为一个 PyTorch 的 Parameter 对象，如下所示：

        Parameter(weight_vector)

        其中 `weight_vector` 是一个维度为 `dimensions` 的 PyTorch 张量(Tensor)。

        提示：你可以使用 `ones(dim)` 来创建一个维度为 `dim` 的全1张量。

        """
        super(PerceptronModel, self).__init__()

        "*** YOUR CODE HERE ***"
        # create a tensor of ones with the given dimensions (1 x dimensions)
        weight_vector = ones(1, dimensions)
        # init weights as a pytorch parameter
        self.w = Parameter(weight_vector)  # Initialize your weights here

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)

        The pytorch function `tensordot` may be helpful here.
        """
        "*** YOUR CODE HERE ***"
        return tensordot(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        dot_product = self.run(x)
        # see lecture 18 slide 12 (Classify with current weights)
        if dot_product >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        You can iterate through DataLoader in order to
        retrieve all the batches you need to train on.

        Each sample in the dataloader is in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.
        """
        with no_grad():
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            "*** YOUR CODE HERE ***"
            converge = False
            while not converge:
                converge = True
                for sample in dataloader:
                    prediction = self.get_prediction(sample["x"])
                    # compare the prediction to the label
                    if prediction != sample["label"]:
                        converge = False
                        self.w += sample["label"] * sample["x"]


class RegressionModel(Module):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    一个神经网络模型，用于近似一个从实数映射到实数的函数。该网络应该足够大，以便能够在区间
    [-2π, 2π]上以合理的精度近似sin(x)函数。
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # super().__init__()
        super(RegressionModel, self).__init__()
        # 定义网络层
        self.fc1 = Linear(1, 64)  # 输入层到隐藏层
        self.fc2 = Linear(64, 64)  # 输入层到隐藏层
        self.fc3 = Linear(64, 1)  # 隐藏层到输出层

    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a tensor of size 1 containing the loss
        """
        "*** YOUR CODE HERE ***"
        predictions = self.forward(x)
        return mse_loss(predictions, y)

    def train(self, dataset):
        """
        Trains the model.

        In order to create batches, create a DataLoader object and pass in `dataset` as well as your required
        batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

        Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.

        Inputs:
            dataset: a PyTorch dataset object containing data to be trained on

        """
        "*** YOUR CODE HERE ***"
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

        # learning_rate = 0.005
        learning_rate = 0.001
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        loss_threshold = 0.01

        while 1:
            for item in dataloader:
                features = item["x"]
                labels = item["label"]
                optimizer.zero_grad()

                loss = self.get_loss(features, labels)

                loss.backward()
                optimizer.step()
            if loss.item() < loss_threshold:
                break

        return self


class DigitClassificationModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        input_size = 28 * 28
        output_size = 10
        "*** YOUR CODE HERE ***"
        # 定义网络层
        self.fc1 = Linear(input_size, 500)  # 输入层到隐藏层
        self.fc2 = Linear(500, 250)  # 输入层到隐藏层
        self.fc3 = Linear(250, 10)  # 隐藏层到输出层

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a tensor with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        """ YOUR CODE HERE """
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a tensor with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss tensor
        """
        """ YOUR CODE HERE """
        prediction = self.run(x)
        return cross_entropy(prediction, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        """ YOUR CODE HERE """
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        learning_rate = 0.001
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        loss_threshold = 0.015
        while True:
            for item in dataloader:
                features = item["x"]
                labels = item["label"]
                optimizer.zero_grad()

                loss = self.get_loss(features, labels)

                loss.backward()
                optimizer.step()
            if loss.item() < loss_threshold:
                break

        return self


class LanguageIDModel(Module):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        super(LanguageIDModel, self).__init__()
        "*** YOUR CODE HERE ***"
        # Initialize your model parameters here
        self.hidden_size = 300  # 可以根据需要调整隐藏层大小

        # 初始网络 f_initial, 将字符数转换为隐藏层大小
        # self.initial_layer = Linear(self.num_chars, self.hidden_size)

        # 递归网络 f,处理隐藏状态和字符输入
        self.char_layer = Linear(self.num_chars, self.hidden_size)
        self.hidden_layer_1 = Linear(self.hidden_size, self.hidden_size)
        # self.hidden_layer_2 = Linear(self.hidden_size, self.hidden_size)
        # self.hidden_layer_3 = Linear(self.hidden_size, self.hidden_size)
        # self.weight_input = Parameter(1, 512)
        # self.bias_input = Parameter(1, 512)
        # self.weight_2_input = Parameter(1, 512)
        # self.bias_2_input = Parameter(1, 512)

        # 输出层,将隐藏层大小转换为语言数
        self.output_layer = Linear(self.hidden_size, len(self.languages))

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        tensor with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a tensor that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single tensor of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a tensor of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # batch_size = xs.shape[1]  # 获取批次大小

        # 计算初始隐藏状态，使用第一个字符和初始层
        # print(f"xs[0]: {xs[0]}")
        # print(f"self.initial_layer(xs[0]): {self.initial_layer(xs[0])}")
        h = self.char_layer(xs[0])
        h = relu(h)

        # 对每个字符进行递归处理，更新隐藏状态
        for x in xs[1:]:
            # 输出形状
            # print(f'Input shape in run: {[x.shape for x in xs]}')

            h = relu(self.hidden_layer_1(h)) + relu(self.char_layer(x))
            # h = relu(self.hidden_layer_2(h))
            # h = relu(self.hidden_layer_3(h))

        # 计算最终输出分数
        # print(f"h: {h}")
        logits = self.output_layer(h)
        # print(f"return shape {logits.shape}")
        return logits

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        logits = self.run(xs)  # 运行模型获取预测值
        return cross_entropy(logits, y)  # 计算交叉熵损失

    def train(self, dataset):
        """
        Trains the model.

        Note that when you iterate through dataloader, each batch will returned as its own vector in the form
        (batch_size x length of word x self.num_chars). However, in order to run multiple samples at the same time,
        get_loss() and run() expect each batch to be in the form (length of word x batch_size x self.num_chars), meaning
        that you need to switch the first two dimensions of every batch. This can be done with the movedim() function
        as follows:

        movedim(input_vector, initial_dimension_position, final_dimension_position)

        For more information, look at the pytorch documentation of torch.movedim()
        """
        "*** YOUR CODE HERE ***"
        learning_rate = 0.00785
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        epoch = 0
        # print("Come to train")
        accuracy = 0
        while accuracy < 0.81:
            # print(f"come to epoch {epoch}")
            total_loss = 0
            for batch in dataloader:
                # print(f"come to batch ")
                xs = batch["x"]  # 获取批次数据中的输入
                y = batch["label"]  # 获取批次数据中的标签
                # print(xs)
                # print(f"Input shape before movedim: {xs.shape}")  # 输出xs的形状
                xs = movedim(xs, 1, 0)  # 按需要交换维度
                # 确保输入的维度是 (length of word, batch_size, num_chars)
                # print(f"Input shape after movedim: {xs.shape}")  # 输出xs的形状
                optimizer.zero_grad()
                loss = self.get_loss(xs, y)
                # print(f"validation accuracy: {dataset.get_validation_accuracy()}")
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                accuracy = dataset.get_validation_accuracy()
            print(f"Accuracy: {accuracy}")


def Convolve(input: tensor, weight: tensor):
    """
    Acts as a convolution layer by applying a 2d convolution with the given inputs and weights.
    DO NOT import any pytorch methods to directly do this, the convolution must be done with only the functions
    already imported.

    There are multiple ways to complete this function. One possible solution would be to use 'tensordot'.
    If you would like to index a tensor, you can do it as such:

    tensor[y:y+height, x:x+width]

    This returns a subtensor who's first element is tensor[y,x] and has height 'height, and width 'width'
    """
    input_tensor_dimensions = input.shape
    weight_dimensions = weight.shape
    Output_Tensor = tensor(())
    "*** YOUR CODE HERE ***"

    "*** End Code ***"
    return Output_Tensor


class DigitConvolutionalModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    This class is a convolutational model which has already been trained on MNIST.
    if Convolve() has been correctly implemented, this model should be able to achieve a high accuracy
    on the mnist dataset given the pretrained weights.


    """

    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        output_size = 10

        self.convolution_weights = Parameter(ones((3, 3)))
        """ YOUR CODE HERE """

    def run(self, x):
        """
        The convolutional layer is already applied, and the output is flattened for you. You should treat x as
        a regular 1-dimentional datapoint now, similar to the previous questions.
        """
        x = x.reshape(len(x), 28, 28)
        x = stack(
            list(map(lambda sample: Convolve(sample, self.convolution_weights), x))
        )
        x = x.flatten(start_dim=1)
        """ YOUR CODE HERE """

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a tensor with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss tensor
        """
        """ YOUR CODE HERE """

    def train(self, dataset):
        """
        Trains the model.
        """
        """ YOUR CODE HERE """
