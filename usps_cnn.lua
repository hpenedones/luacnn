
require "torch"
require "image"
require "nn"
require "math"

-- global variables

images_set={}
windows={}

name_format = "usps_%d.png"

classes = {1,2,3,4,5,6,7,8,9,10} -- indices in torch5/lua start at 1, not at zero
classes_names = {'0','1','2','3','4','5','6','7','8','9'}

ncols = 33
nrows = 34
sub_image_height, sub_image_width = 16, 16
train_size=1000
total_examples_per_class=1100

inputs=sub_image_height*sub_image_width

learningRate = 0.01
maxIterations = 100000

GUI_ON = false

print("OK")

function load_data_from_disk(folder)
    for i=1,10 do
        local filename = string.format(name_format,i-1)
        images_set[i] = image.loadPNG(folder .. filename,1)    -- images_set is global
        images_set[i]:resize(images_set[i]:size(2), images_set[i]:size(3))
        if GUI_ON then
            image.display(images_set[i])
        end
    end
end



-- returns the tensor pointing to sample example_id
-- note that this function knows about global variable images_set and the sizes of subimages 
function get_example(class, example_id)
    local image = images_set[class]
    local example_row = 1 + (example_id-1) % nrows 
    local example_col = 1 + math.floor((example_id-1) / nrows)
    -- print('class:', class, 'example_id:', example_id, 'len of images_set: ', #images_set)
    -- print(example_row, example_col, 
    --        (example_row-1)*sub_image_height + 1, example_row * sub_image_height, 
    --        (example_col-1)*sub_image_width  + 1, example_col * sub_image_width
    --        )

    local ret = image:sub( 
            (example_row-1)*sub_image_height + 1, example_row * sub_image_height,
            (example_col-1)*sub_image_width  + 1, example_col * sub_image_width
            )
    return ret:reshape(1, sub_image_width, sub_image_height);
end

-- returns a dataset 
function create_dataset(classes, first_index, last_index)

    local nsamples_per_class = (last_index - first_index + 1) 

    local dataset={};
    function dataset:size() return #classes*nsamples_per_class end
    

    local index = 0

    for c=1,#classes do
       for i=first_index,last_index do
            local cc=classes[c]
            local input  = get_example(cc, i)
            index = index + 1
            dataset[index] = {input, c}
        end
    end



    return dataset
end

-- here we set up the architecture of the neural network
function create_network(nb_outputs)

    local ann = nn.Sequential();  -- make a multi-layer structure
                                                -- 16x16x1                
    ann:add(nn.SpatialConvolution(1,6,5,5))   -- becomes 12x12x6
    ann:add(nn.SpatialSubSampling(6,2,2,2,2)) -- becomes  6x6x6 

    ann:add(nn.Reshape(6*6*6))
    ann:add(nn.Tanh())
    ann:add(nn.Linear(6*6*6,nb_outputs))
    ann:add(nn.LogSoftMax())
    
    return ann
end

-- train a Neural Netowrk
function train_network( network, dataset)
        
    print( "Training the network" )
    local criterion = nn.ClassNLLCriterion()
    
    for iteration=1,maxIterations do
        local index = math.random(dataset:size()) -- pick example at random
        local input = dataset[index][1]        
        local output = dataset[index][2]
        if iteration%2000==0 then
            print("iteration: "..iteration, "index: "..index, "input.size:", input:size(), "output: ", output)
        end
        inp=network:forward(input)
        criterion:forward(inp, output)

        network:zeroGradParameters()
        network:backward(input, criterion:backward(network.output, output))
        network:updateParameters(learningRate)
    end
    
end



function test_predictor(predictor, test_dataset, classes, classes_names)

    local mistakes = 0
    local tested_samples = 0
    
    print( "----------------------" )
    print( "Index Label Prediction" )
    for i=1,test_dataset:size() do

        local input  = test_dataset[i][1]
        local class_id = test_dataset[i][2]
    
        local responses_per_class  =  predictor:forward(input) 
        local probabilites_per_class = torch.exp(responses_per_class)
        local probability, prediction = torch.max(probabilites_per_class, 1) 

            
        if prediction[1] ~= class_id then
            mistakes = mistakes + 1
            local label = classes_names[ classes[class_id] ]
            local predicted_label = classes_names[ classes[prediction[1] ] ]
            print(i , label , predicted_label )
        end

        tested_samples = tested_samples + 1
    end

    local test_err = mistakes*100/tested_samples
    print ( "Test error " .. test_err .. "% ( " .. mistakes .. " out of " .. tested_samples .. " )")

end


-- main routine
function main()

    data_folder = arg[1] or "data" -- pass absolute path where data is from the command line

    load_data_from_disk(data_folder .. "/")

    local training_dataset = create_dataset(classes, 1, train_size)
    local testing_dataset   = create_dataset(classes, train_size + 1, total_examples_per_class)
    local network = create_network(#classes)

    print("training_dataset:", training_dataset:size())
    train_network(network, training_dataset)
    
    test_predictor(network, testing_dataset, classes, classes_names)

end


main()
















