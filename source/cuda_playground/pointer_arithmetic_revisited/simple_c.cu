//
// Created by andreas on 03.01.22.
//

#include <iostream>


template <typename T>
void allocate_1d_memory(T *& pointer_to_T, const size_t number_of_elements)
{
	pointer_to_T = static_cast<T*>(calloc(number_of_elements, sizeof(T)));
}

template <typename T>
void free_1d_memory(T *& pointer_to_T)
{
	free(pointer_to_T);
	pointer_to_T = nullptr;

}

template <typename T>
void allocate_2d_memory(T **& pointer_to_pointer_to_T, const size_t number_of_first_index_elements, const size_t number_of_second_index_elements)
{
	pointer_to_pointer_to_T = static_cast<T**>(calloc(number_of_first_index_elements, sizeof(T*)));
	for(size_t i = 0; i<number_of_first_index_elements; ++i)
	{
		pointer_to_pointer_to_T[i] = static_cast<T*>(calloc(number_of_second_index_elements, sizeof(T)));
	}
}

template <typename T>
void free_2d_memory(T *& pointer_to_pointer_to_T, const size_t number_of_first_index_elements)
{
	for(int i = 0; i < number_of_first_index_elements; i++)
	{
		free(pointer_to_pointer_to_T[i]);
	}
	free(pointer_to_pointer_to_T);

}




int main()
{
	int * p_int = nullptr;
	size_t number_of_elements = 5;
	for(int j =0; j <3; ++j) {
		allocate_1d_memory(p_int, number_of_elements);
		for (int i = 0; i < 5; ++i) {
			p_int[i] = 10.1f + i * 10.f;
		}

		for (int i = 0; i < 5; ++i) {
			std::cout << p_int[i] << std::endl;
		}
		free_1d_memory(p_int);
	}

	int ** p_p_int;
	size_t number_of_sub_elements = 3;
	allocate_2d_memory(p_p_int, number_of_elements, number_of_sub_elements);
	for(int i = 0; i < number_of_elements; i++)
	{
		for(int j = 0; j < number_of_sub_elements; j++) {
			p_p_int[i][j] = i+j;
		}
	}
	for(int i = 0; i < number_of_elements; i++)
	{
		for(int j = 0; j < number_of_sub_elements; j++) {
			std::cout << p_p_int[i][j] << std::endl;
		}
	}

	free_2d_memory(p_p_int, number_of_elements);
	std::cout << "Hallo" << std::endl;
	return 0;
}