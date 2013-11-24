require "torch"
require "image"
require "nn"
require "math"
require "paths"

image_width = 66
learningRate = 0.005
maxIterations = 10000

function loadBMP(fn)
    local s=io.open(fn,"rb"):read("*all")
    local d={}
    local i=0
    for i=1,128*128 do d[i]=s:byte(1078+i)/255 end
    local ret = torch.Tensor(d)
    ret=image.scale(ret:resize(128, 128), image_width, image_width)
    return ret:resize(1, image_width, image_width)
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
    local test={}
    local train={}
    local images = {}
    local image_count = {}
    for i=1, #names do
        images[i]=torch.Tensor((image_width+1)*10, (image_width+1)*10)
        images[i]:fill(0)
        image_count[i]=0
    end
    local test_tag=torch.Tensor(5,5)
    test_tag:fill(1)
    for i=1, #all_index do
        local files=all_index[i]
        for index=1, #files do
            local data=dataset[files[index]]
            local c=image_count[data[2]]
            local x,y=c%10,0
            y=(c-x)/10
            --print('class:'..data[2], 'image count:'..c, 'col:'..x, 'row:'..y)
            if y<10 then
                x, y=x*(image_width+1), y*(image_width+1)
                local image=images[data[2]]
                image[{{y+1, y+image_width}, {x+1, x+image_width}}]=data[1]
                if index>=#files-test_num then -- for test, show a tag
                    image[{{y+1, y+5}, {x+1, x+5}}]=test_tag
                end
                image_count[data[2]]=c+1
            end
            if index < #files-test_num then
                train[#train+1]=data
            else
                test[#test+1]=data
            end
        end
    end
    for i=1, #names do
        image.save(names[i]..'.jpg', images[i])
    end
    return names, train, test
end

-- here we set up the architecture of the neural network
function create_network(size, nb_outputs)
    print("create_network: input image size="..size..",", "output number:"..nb_outputs)
    local ann = nn.Sequential()  -- make a multi-layer structure
    local filter_size, filter_num, subsample_size, subsample_step=3, 15, 4, 4
    local filter_size2, filter_num2, subsample_size2, subsample_step2=3, 24, 2, 2
                                                -- 16x16x1
    print('  1@'..size..'x'..size)
    ann:add(nn.SpatialConvolution(1, filter_num, filter_size, filter_size))   -- becomes 12x12x6
    local l2size=size-filter_size+1
    print(' -> '..filter_num..'@'..l2size..'x'..l2size)
    ann:add(nn.SpatialSubSampling(filter_num, subsample_size, subsample_size, subsample_step, subsample_step)) -- becomes  6x6x6 
    local unit_size=(l2size-subsample_size)/subsample_step+1
    print(' -> '..filter_num..'@'..unit_size..'x'..unit_size)
    -- ann:add(nn.Reshape(filter_num, unit_size, unit_size))
    ann:add(nn.SpatialConvolution(filter_num, filter_num*filter_num2, filter_size2, filter_size2))   -- becomes 12x12x6
    local l2size=unit_size-filter_size2+1
    filter_num=filter_num*filter_num2
    print(' -> '..filter_num..'@'..l2size..'x'..l2size)
    ann:add(nn.SpatialSubSampling(filter_num, subsample_size2, subsample_size2, 
        subsample_step2, subsample_step2)) -- becomes  6x6x6 
    unit_size=(l2size-subsample_size2)/subsample_step2+1
    print(' -> '..filter_num..'@'..unit_size..'x'..unit_size)
    local unit_num=filter_num*unit_size*unit_size
    ann:add(nn.Reshape(unit_num))
    print(' -> '..unit_num)
    ann:add(nn.Tanh())
    ann:add(nn.Linear(unit_num, nb_outputs))
    ann:add(nn.LogSoftMax())
    print(' -> '..nb_outputs)
    return ann
end

-- train a Neural Netowrk
function train_network( network, dataset)
        
    print( "Training the network" )
    local criterion = nn.ClassNLLCriterion()
    
    for iteration=1,maxIterations do
        local index = math.random(#dataset) -- pick example at random
        local input = dataset[index][1]        
        local output = dataset[index][2]
        if iteration%2000==0 then
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
    print( "\tIndex Label Prediction" )
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
            print("", "error:", test_dataset[i][3], i, label, predicted_label )
        end

        tested_samples = tested_samples + 1
    end

    local test_err = mistakes*100/tested_samples
    print ("Test error " .. test_err .. "% ( " .. mistakes .. " out of " .. tested_samples .. " )")

end


-- main routine
function main()

    local classes_names, training_dataset, testing_dataset = create_dataset('fv-images/', 7)
    print("classes number:", #classes_names)
    print("training_dataset:", #training_dataset)
    print("testing_dataset :", #testing_dataset)

    local network = create_network(image_width, #classes_names)

    train_network(network, training_dataset)
    
    test_predictor(network, testing_dataset, classes_names)

end


main()

