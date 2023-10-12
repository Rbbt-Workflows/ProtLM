require 'rbbt-util'
require 'rbbt/workflow'
require 'rbbt/util/python'

Misc.add_libdir if __FILE__ == $0

#require 'rbbt/sources/ProtLM'

module ProtLM
  extend Workflow

  input :checkpoint, :string, "Checkpoint or model name", 'nferruz/ProtGPT2'
  input :training, :file, "File with training", nil, :noload => true, :nofile => true
  input :validation, :file, "File with validation", nil, :noload => true, :nofile => true
  task :train_CLM => :text do |checkpoint,training,validation|
    script = Rbbt.scripts.clm.set_extension('py').find

    training = File.expand_path(training)
    validation = File.expand_path(validation) if validation

    output = file('output')
    options = {
      model_name_or_path: checkpoint,
      train_file: training,
      validation_file: validation,
      output_dir: output,
      do_train: true,
      do_eval: true,
      num_train_epochs: 1,
      learning_rate: 5e-07,
      per_device_train_batch_size: 1,
      per_device_eval_batch_size: 1,
      logging_strategy: 'epoch',
      evaluation_strategy: 'steps',
      eval_steps: 250,
      load_best_model_at_end: true,
      overwrite_output_dir: true
    }
    Misc.in_dir file('work') do
      CMD.cmd("python #{script}", options.merge(add_option_dashes: true))
    end
  end

  input :checkpoint, :string, "Checkpoint or model name"
  input :training, :file, "File with training", nil, :noload => true
  input :validation, :file, "File with validation", nil, :noload => true
  task :train_MLM => :string do |checkpoint,training,validation|
    script = Rbbt.scripts.mlm.set_extension('py').find

    output = file('output')
    options = {
      model_name_or_path: checkpoint,
      train_file: training,
      validation_file: validation,
      output_dir: output,
      num_train_epochs: 100,
      per_device_train_batch_size: 1,
      per_device_eval_batch_size: 1,
      do_train: true,
      do_eval: true,
      evaluation_strategy: 'steps',
      eval_steps: 250,
      load_best_model_at_end: true,
      overwrite_output_dir: true,
      line_by_line: true
    }
    Misc.in_dir file('work') do
      CMD.cmd_log("python #{script}", options.merge(add_option_dashes: true))
    end
    "DONE"
  end


  dep :train_CLM
  input :number_of_seq, :integer, "Number of sequences to generate", 100
  task :generate_sequences => :array do |number_of_seq|
    checkpoint = step(:train_CLM).file('output')
    sequences = []
    RbbtPython.add_path Rbbt.python.find
    RbbtPython.run do
      pyimport :numpy, as: :np
      pyimport :torch
      pyimport :transformers

      tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
      begin
        model = transformers.AutoModelForCausalLM.from_pretrained(checkpoint).to('cuda:0')
      rescue
        model = transformers.AutoModelForCausalLM.from_pretrained(checkpoint)
      end

      number_of_seq.times do
        generated_text = model.generate(
          bos_token_id: tokenizer.bos_token_id,
          do_sample: true,
          max_length: 400,
          top_k: 50,
          top_p: 0.95,
          temperature: 1.0,
          no_repeat_ngram_size: 2
        )

        sequence = tokenizer.decode(np.squeeze(generated_text), skip_special_tokens: true)	
        sequence = sequence.gsub("\n",'')
        sequences << sequence
      end
    end

    sequences
  end

  input :clm_training, :file, "File with CLM training", nil, :noload => true, :nofile => true
  input :clm_validation, :file, "File with CLM  validation", nil, :noload => true, :nofile => true
  input :mlm_training, :file, "File with MLM training", nil, :noload => true, :nofile => true
  input :mlm_validation, :file, "File with MLM  validation", nil, :noload => true, :nofile => true
  dep :generate_sequences, :training => :clm_training, :validation => :clm_validation
  dep :train_MLM, :training => :mlm_training, :validation => :mlm_validation
  task :evaluate_sequences => :tsv do
  end

  dep_task :pilot_clm, ProtLM, :generate_sequences, 
    :training => Rbbt.data["fine-tuning-data-protgpt2/training.txt"].find,
    :validation => Rbbt.data["fine-tuning-data-protgpt2/validation.txt"].find
end

#require 'ProtLM/tasks/basic.rb'

#require 'rbbt/knowledge_base/ProtLM'
#require 'rbbt/entity/ProtLM'

