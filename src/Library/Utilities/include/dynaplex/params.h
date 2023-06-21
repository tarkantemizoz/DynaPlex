#pragma once
#include <memory>
#include <string>
#include <variant>
#include <vector>
#include "json_fwd.h"
#include <concepts>
 
namespace DynaPlex {
	class Params;

	template<typename T>
	concept ConvertibleFromParams = requires(Params& p) {
		{ T(p) };
	};

	template<typename T>
	concept ConvertibleFromParamsVec = requires(T a, typename T::value_type val) {
		typename T::value_type;
		{ T() };
        requires ConvertibleFromParams<typename T::value_type>;
		a.push_back(val);
	};

	class Params
	{


	public:
		using Int64Vec = std::vector<int64_t>;
		using DoubleVec = std::vector<double>;
		using StringVec = std::vector<std::string>;
		using ParamsVec = std::vector<Params>;
		using DataType = std::variant<bool,std::nullptr_t, int64_t, double, std::string,DynaPlex::Params, Int64Vec,DoubleVec,StringVec,ParamsVec>;
		using TupleList = std::initializer_list< std::tuple<std::string, DataType>>;
	    
		
	    Params();		
		Params(const Params& other);
		Params& operator=(const Params& other);
		~Params();
		Params(Params&& other) noexcept;
		Params& operator=(Params&& other) noexcept;


		Params(TupleList list);
		

		void Add(std::string s, const Params& vec);
		void Add(std::string s, int val);
		void Add(std::string s, int64_t val);
		void Add(std::string s, bool val);
		void Add(std::string s, std::string val);
		void Add(std::string s, double val);
		void Add(std::string s, const Int64Vec& vec);
		void Add(std::string s, const StringVec& vec);
		void Add(std::string s, const DoubleVec& vec);
		void Add(std::string s, const ParamsVec& vec);



		void GetInto(const std::string& key, Params& params) const;
		void GetInto(const std::string& key, int64_t& out_val) const;
		void GetInto(const std::string& key, std::string& out_val) const;
		void GetInto(const std::string& key, int& out_val) const;
		void GetInto(const std::string& key, bool& out_val) const;
		void GetInto(const std::string& key, double& out_val) const;
		void GetInto(const std::string& key, Int64Vec& out_val) const;
		void GetInto(const std::string& key, StringVec& out_val) const;
		void GetInto(const std::string& key, DoubleVec& out_val) const;
		void GetInto(const std::string& key, ParamsVec& out_val) const;
		void GetInto(const std::string& key, std::vector<int>& out_val) const;
		
		
		template<ConvertibleFromParams T>
		void GetInto(const std::string& key, T& out_val) const {
			Params params;
			GetInto(key, params);
			out_val = T(params);
		}



		template<ConvertibleFromParamsVec T>
		void GetInto(const std::string& key, T& out_val) const {
			out_val.clear();

			ParamsVec paramsVec;
			GetInto(key, paramsVec);

			for (Params& p : paramsVec) {
				out_val.push_back(typename T::value_type(p));
			}
		}

		void SaveToFile(const std::string &filename) const;
		static Params LoadFromFile(const std::string &filename);


		void Print() const;

	protected:
		Params(nlohmann::ordered_json json);
		nlohmann::ordered_json ToJson() const;

	private:
		struct Impl;
		std::unique_ptr<Impl> pImpl;

	};
}//namespace DynaPlex