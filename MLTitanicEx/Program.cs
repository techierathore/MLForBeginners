using Microsoft.ML;
using Microsoft.ML.Data;
using System;

namespace MLTitanicEx
{
    class Program
    {
        static void Main(string[] args)
        {
            // Steps 
            var mlContext = new MLContext(seed: 1); 
            // We are using seed to get consistent data every time. when we split the data it will be ramdom but same random eveytime 

            //1-Load Data 
            var data = mlContext.Data.LoadFromTextFile<Passenger>("train.csv", 
                separatorChar:',', hasHeader:true);
            //2-Split Data 
            var testTrainSplit = mlContext.Data.TrainTestSplit(data);
            var testData = testTrainSplit.TestSet;
            var traindata = testTrainSplit.TrainSet;
            //3-Transform/clean the data (conveting text to number etc)
            var dataPipeline = mlContext.Transforms.Categorical.OneHotEncoding(nameof(Passenger.Sex))
                .Append(mlContext.Transforms.ReplaceMissingValues(nameof(Passenger.Age),
                replacementMode: Microsoft.ML.Transforms.MissingValueReplacingEstimator.ReplacementMode.Mean))
                .Append(mlContext.Transforms.Concatenate("Features", "Pclass","Sex","Age"));
            // Order of Fields doen;t matter in Concatenate 
            //4-Train the model 
            var trainingPipeline = dataPipeline.Append(
                mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(nameof(Passenger.Survived)));
            var trainedModel = trainingPipeline.Fit(traindata);
            //5-evaluate the Model 
            var predictions = trainedModel.Transform(testData);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, nameof(Passenger.Survived));

            //Save the model
            mlContext.Model.Save(trainedModel,traindata.Schema,"TrainedModel.zip");
        }
    }

    /// <summary>
    /// Note for Practice
    /// Steps to increase the metrics score 
    /// 1) Add more (we need check)
    /// 2) Take Seperate test data 
    /// 3) Use different algorothim 
    /// 4) Use all the features [simplest]
    /// </summary>



    public class Passenger
    {
        //PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
        [LoadColumn(1)]
       public bool Survived { get; set; }
        [LoadColumn(2)]
        public float  Pclass { get; set; } // Always use float for numbers in ML.NET (Alex's tricks)
        [LoadColumn(4)]
        public string Sex { get; set; }
        [LoadColumn(5)]
        public float Age { get; set; }
    }

}
