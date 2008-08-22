% based off of tutorial 
% http://www.trapexit.org/Erlang_and_Neural_Networks

-module(ann).
-export([perceptron/4, connect/2]).


perceptron(Weights, Inputs, Sensitivities, Stale_inputs) ->
  Sigmoid = fun(X) -> 1/(1+math:exp(-X)) end,
  Sigmoid_deriv = fun(X) -> math:exp(-X)/(1+math:exp(-2*X)) end,

  receive
    {stimulate, Input} ->
      % add input to inputs
      New_stale_inputs = delete_input(Stale_inputs, Input),
      New_inputs = replace_input(Inputs, Input),
      
      {In_PID, In_Val, Expected_Val} = Input,
      ann_grapher ! {stimulate, In_PID, self(), In_Val},

      case New_stale_inputs of
        % calculate perceptron output
        [] -> Output = feed_forward(Sigmoid, Weights, 
                          convert_to_input_values(New_inputs)),
  
        % stimulate connected perceptrons
        case Sensitivities of
          [] -> 
            ann_grapher ! {output, self(), Output},
            io:format("Output (~w): ~w~n", [self(), Output]),
            case Expected_Val of
              null -> ok;
              E -> self() ! {learn, {self(), E}}
            end;
          S -> lists:foreach(fun(Output_PID) ->
                Output_PID ! {stimulate, {self(), Output, Expected_Val}}
              end,
              convert_to_keys(S))
        end,
        perceptron(Weights, New_inputs, Sensitivities, New_inputs);

        New_stale -> perceptron(Weights, New_inputs, Sensitivities, New_stale)
      end;

    {learn, Backprop} ->
    
      {Back_source, Back_value} = Backprop,
      ann_grapher ! {backprop, Back_source, self(), Back_value},

      Learning_rate = 0.5,
      New_sensitivities = add_sensitivity(Sensitivities, Backprop),
      Output = feed_forward(Sigmoid, Weights, 
                  convert_to_input_values(Inputs)),
      Deriv = feed_forward(Sigmoid_deriv, Weights, 
                  convert_to_input_values(Inputs)),
      Sensitivity = calculate_sensitivities(Backprop, Inputs,
                  New_sensitivities, Output, Deriv),

      Weight_adjustments = lists:map(fun(Input) ->
                              Learning_rate * Sensitivity * Input
                            end,
                            convert_to_input_values(Inputs)),
      New_weights = vector_map(fun(W, D) -> W+D end, Weights,
                      Weight_adjustments),

      vector_map(fun(Weight, Input_PID) ->
          Input_PID ! {learn, {self(), Sensitivity * Weight}}
        end,
        New_weights,
        convert_to_input_keys(Inputs)),

      perceptron(New_weights, Inputs, New_sensitivities, Stale_inputs);

    {connect_to_output, Receiver_PID} ->
      Combined_sensitivities = 
        [{Receiver_PID, 0.5} | Sensitivities],
      perceptron(Weights, Inputs, Combined_sensitivities, Stale_inputs);

    {connect_to_input, Sender_PID} ->
      Combined_input = [{Sender_PID, 0.5, null} | Inputs],
      New_stale_input = [{Sender_PID, 0.5, null} | Stale_inputs],
      perceptron([0.5 | Weights], Combined_input, Sensitivities, New_stale_input);

    {pass, Input_value, Expected_Output} ->
      lists:foreach(fun(Output_PID) ->
          Output_PID ! {stimulate, {self(), Input_value, Expected_Output}}
        end,
        convert_to_keys(Sensitivities)),
      perceptron(Weights, Inputs, Sensitivities, Stale_inputs)

  end.

feed_forward(Func, Weights, Inputs) -> 
  Func(dot_prod(Weights, Inputs)).

add_sensitivity([], _Backprop) -> [];
add_sensitivity(Sensitivities, Backprop) -> 
  replace_sensitivity_input(Sensitivities, Backprop).

calculate_sensitivities(_Backprop, [], 
  _Sensitivities, _Output, _Deriv) ->
  null;
calculate_sensitivities({_, Training_value}, _Inputs, 
  [], Output, Deriv) ->
  (Training_value - Output) * Deriv;
calculate_sensitivities(_Backprop, _Inputs, 
  Sensitivities, _Output, Deriv) ->
  Deriv * lists:sum(convert_to_values(Sensitivities)).

connect(Sender, Receiver) ->
  ann_grapher ! {connect, Sender, Receiver},
  Sender ! {connect_to_output, Receiver},
  Receiver ! {connect_to_input, Sender}.

replace_sensitivity_input(Inputs, Input) ->
  {Input_PID, _} = Input,
  lists:keyreplace(Input_PID, 1, Inputs, Input).

replace_input(Inputs, Input) ->
  {Input_PID, _, _} = Input,
  lists:keyreplace(Input_PID, 1, Inputs, Input).

delete_input(Inputs, Input) ->
  {Input_PID, _, _} = Input,
  lists:keydelete(Input_PID, 1, Inputs).


convert_to_input_values(Input_list) ->
  lists:map(fun({_, Val, _}) -> Val end, Input_list).

convert_to_input_keys(Input_list) ->
  lists:map(fun({Key, _, _}) -> Key end, Input_list).

convert_to_values(Tuple_list) ->
  lists:map(fun({_, Val}) -> Val end, Tuple_list).

convert_to_keys(Tuple_list) ->
  lists:map(fun({Key, _}) -> Key end, Tuple_list).

dot_prod(X, Y) -> dot_prod(0, X, Y).

dot_prod(Acc, [], []) -> Acc;
dot_prod(Acc, [X1|X], [Y1|Y]) ->
  dot_prod(X1*Y1 + Acc, X, Y).

vector_map(Func, X, Y) ->
  vector_map([], Func, X, Y).

vector_map(Acc, _Func, [], []) ->
  lists:reverse(Acc);
vector_map(Acc, Func, [Xh | Xt], [Yh | Yt]) ->
  vector_map([Func(Xh, Yh)|Acc], Func, Xt, Yt).
