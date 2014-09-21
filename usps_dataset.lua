
require "torch"
require "image"
require "math"

-- global variables

images_set={}
name_format = "usps_%d.png"

classes = {1,2,3,4,5,6,7,8,9,10} -- indices in torch/lua start at 1, not at zero
classes_names = {'0','1','2','3','4','5','6','7','8','9'}

ncols = 33
nrows = 34
sub_image_height, sub_image_width = 16, 16
train_size=1000
total_examples_per_class=1100

inputs=sub_image_height*sub_image_width

function load_data_from_disk(folder)
   
   for i=1,10 do
      local filename = string.format(name_format,i-1)
      images_set[i] = image.loadPNG(folder .. filename,1)      -- images_set is global
      images_set[i]:resize(images_set[i]:size(2), images_set[i]:size(3))
   end
end

-- returns the tensor pointing to sample example_id
-- note that this function knows about global variable images_set and the sizes of subimages 
function get_example(class, example_id)

   local image = images_set[class]
   
   local example_row = 1 + (example_id-1) % nrows 
   local example_col = 1 + math.floor((example_id-1) / nrows)
   
   local ex = image:sub( 
                    (example_row-1)*sub_image_height + 1,
                    example_row * sub_image_height,
                    (example_col-1)*sub_image_width + 1,
                    example_col * sub_image_width
                     )
   return ex:reshape(1, sub_image_width, sub_image_height)
end

-- returns a dataset 
function create_dataset(classes, first_index, last_index)

   local nsamples_per_class = (last_index - first_index + 1) 
      
   local dataset={};
   function dataset:size() return #classes*nsamples_per_class end
   
   local index = 0
   
   for c=1,#classes do
      for i=first_index,last_index do
         local input  = get_example(classes[c], i)
         index = index + 1
         dataset[index] = {input, c}
      end
   end
   
   return dataset
end



load_data_from_disk("data/")
           
local training_dataset = create_dataset(classes, 1, train_size)
local testing_dataset   = create_dataset(classes, train_size + 1, total_examples_per_class)
        


return training_dataset, testing_dataset, classes, classes_names















