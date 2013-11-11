require "torch"
require "image"
require "nn"
require "math"
require "paths"

function loadBMP(fn)
    local s=io.open(fn,"rb"):read("*all")
    local d={}
    local i=0
    for i=1,128*128 do d[i]=s:byte(1078+i)/255 end
    local ret = torch.Tensor(d)
    return ret:resize(1, 128, 128)
end

--print(loadBMP("/media/d/work/fv/Camera/fv-images/norm/1_1.bmp"))

-- returns a dataset 
function create_dataset(path, test_num)

    local dataset={}
    local names={}
    local all_index={}
    local index=0
    for f in paths.files(path) do
        local fn=path..'/'..f
        if f:sub(-4)=='.bmp' and paths.filep(fn) then
            local f=f:sub(1, -5)
            local name = f:match(".*_")
            name=name:sub(1, -2)
            local i=names[name]
            if i==nil then
                i=#names+1
                names[name]=i
                names[i]=name 
            end
            index=index+1
            dataset[index]={loadBMP(fn), i, f}
            if all_index[i]==nil then 
                all_index[i]={index}
            else
                local ii=all_index[i]
                ii[#ii+1]=index
            end
        end
    end
    test={}
    train={}
    for i=1, #all_index do
        local files=all_index[i]
        for index=1, #files do
            local data=dataset[files[index]]
            if index < #files-test_num then
                train[#train+1]=data
            else
                test[#test+1]=data
            end
        end
    end
    return names, train, test
end

-- here we set up the architecture of the neural network
function create_network(size, nb_outputs)
    print("create_network: input image size="..size..",", "output number:"..nb_outputs)
    local ann = nn.Sequential()  -- make a multi-layer structure
    local filter_size, filter_num, subsample_size, subsample_step=15, 64, 6, 6
                                                -- 16x16x1
    ann:add(nn.SpatialConvolution(1, filter_num, filter_size, filter_size))   -- becomes 12x12x6
    ann:add(nn.SpatialSubSampling(filter_num, subsample_size, subsample_size, subsample_step, subsample_step)) -- becomes  6x6x6 
    local l2size=size-filter_size+1
    local unit_size=(l2size-subsample_size)/subsample_step+1
    local unit_num=filter_num*unit_size*unit_size
    print('  1@'..size..'x'..size..
        ' -> '..filter_num..'@'..l2size..'x'..l2size..
        ' -> '..filter_num..'@'..unit_size..'x'..unit_size..
        ' -> '..nb_outputs)
    ann:add(nn.Reshape(unit_num))
    ann:add(nn.Tanh())
    ann:add(nn.Linear(unit_num, nb_outputs))
    ann:add(nn.LogSoftMax())
    
    return ann
end

-- train a Neural Netowrk
function train_network( network, dataset)
        
    local learningRate = 0.01
    local maxIterations = 10000

    print( "Training the network" )
    local criterion = nn.ClassNLLCriterion()
    
    for iteration=1,maxIterations do
        local index = math.random(#dataset) -- pick example at random
        local input = dataset[index][1]        
        local output = dataset[index][2]
        if iteration%1000==0 then
            print("\titeration: "..iteration.."/"..maxIterations)
        end
        local inp=network:forward(input)
        --if iteration==1 then print(input:size(), output, inp:size(), #dataset) end
        criterion:forward(inp, output)

        network:zeroGradParameters()
        network:backward(input, criterion:backward(network.output, output))
        network:updateParameters(learningRate)
    end
    
end



function test_predictor(predictor, test_dataset, classes_names)

    local mistakes = 0
    local tested_samples = 0
    
    print( "----------------------" )
    print( "Index Label Prediction" )
    for i=1, #test_dataset do

        local input  = test_dataset[i][1]
        local class_id = test_dataset[i][2]
    
        local responses_per_class  =  predictor:forward(input) 
        local probabilites_per_class = torch.exp(responses_per_class)
        local probability, prediction = torch.max(probabilites_per_class, 1) 
            
        if prediction[1] ~= class_id then
            mistakes = mistakes + 1
            local label = classes_names[ class_id ]
            local predicted_label = classes_names[ prediction[1] ]
            print("", "error:", i, label, predicted_label )
        end

        tested_samples = tested_samples + 1
    end

    local test_err = mistakes*100/tested_samples
    print ("Test error " .. test_err .. "% ( " .. mistakes .. " out of " .. tested_samples .. " )")

end


-- main routine
function main()

    local classes_names, training_dataset, testing_dataset = create_dataset('/media/d/work/fv/Camera/fv-images/norm/', 2)
    print("classes number:", #classes_names)
    print("training_dataset:", #training_dataset)
    print("testing_dataset :", #testing_dataset)

    local network = create_network(128, #classes_names)

    train_network(network, training_dataset)
    
    test_predictor(network, testing_dataset, classes_names)

end


main()

