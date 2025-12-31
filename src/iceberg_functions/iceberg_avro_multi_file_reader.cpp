#include "iceberg_avro_multi_file_reader.hpp"
#include "duckdb/common/exception.hpp"

namespace duckdb {

unique_ptr<MultiFileReader> IcebergAvroMultiFileReader::CreateInstance(const TableFunction &table) {
	auto reader = make_uniq<IcebergAvroMultiFileReader>();
	// Extract file_infos from function_info if available
	if (table.function_info) {
		auto &info = table.function_info->Cast<IcebergAvroScanInfo>();
		reader->file_infos = info.files;  // Copy, not move - CreateInstance may be called multiple times
	}
	return reader;
}

shared_ptr<MultiFileList> IcebergAvroMultiFileReader::CreateFileList(ClientContext &context,
                                                                     const vector<string> &paths,
                                                                     const FileGlobInput &glob_input) {
	vector<OpenFileInfo> open_files;
	for (idx_t i = 0; i < paths.size(); i++) {
		auto &path = paths[i];

		// Check if we have pre-populated OpenFileInfo with extended_info from function_info
		if (i < file_infos.size() && file_infos[i].extended_info) {
			// Copy the pre-populated OpenFileInfo (contains file_size, etag, etc.)
			// Must copy, not move - CreateFileList may be called multiple times
			open_files.push_back(file_infos[i]);
		} else {
			open_files.emplace_back(path);
		}

		if (!open_files.back().extended_info) {
			open_files.back().extended_info = make_shared_ptr<ExtendedOpenFileInfo>();
		}
		open_files.back().extended_info->options["validate_external_file_cache"] = Value::BOOLEAN(false);
		open_files.back().extended_info->options["force_full_download"] = Value::BOOLEAN(true);
	}
	return make_uniq<SimpleMultiFileList>(std::move(open_files));
}

} // namespace duckdb
