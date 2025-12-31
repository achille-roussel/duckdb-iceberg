//===----------------------------------------------------------------------===//
//                         DuckDB
//
// iceberg_avro_multi_file_reader.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/multi_file/multi_file_reader.hpp"
#include "duckdb/function/table_function.hpp"

namespace duckdb {

//! Custom TableFunctionInfo to pass OpenFileInfo through the bind pipeline
struct IcebergAvroScanInfo : public TableFunctionInfo {
	vector<OpenFileInfo> files;

	explicit IcebergAvroScanInfo(vector<OpenFileInfo> files_p) : files(std::move(files_p)) {
	}
};

struct IcebergAvroMultiFileReader : public MultiFileReader {
	//! File infos passed via function_info, containing extended_info with file_size etc.
	vector<OpenFileInfo> file_infos;

	shared_ptr<MultiFileList> CreateFileList(ClientContext &context, const vector<string> &paths,
	                                         const FileGlobInput &glob_input) override;

	static unique_ptr<MultiFileReader> CreateInstance(const TableFunction &table);
};

} // namespace duckdb
