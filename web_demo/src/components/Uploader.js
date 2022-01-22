/* eslint-disable import/no-anonymous-default-export */
import React from 'react'
import { PageHeader, Upload, message } from 'antd';
import { FileAddFilled } from '@ant-design/icons';

const { Dragger } = Upload;

export default ({ setUpload }) => {
    const fileProps = {
        name: 'file',
        multiple: false,
        maxCount: 1,
        // action: 'https://www.mocky.io/v2/5cc8019d300000980a055e76',
        // onChange(info) {
        //     const { status } = info.file;
        //     if (status !== 'uploading') {
        //         console.log(info.file, info.fileList);
        //     }
        //     if (status === 'done') {
        //         message.success(`${info.file.name} file uploaded successfully.`);
        //         localStorage.setItem("uploadedFiles", [info.file])
        //         setUpload(true)
        //     } else if (status === 'error') {
        //         message.error(`${info.file.name} file upload failed.`);
        //     }
        // },
        onDrop(e) {
            console.log('Dropped files', e.dataTransfer.files);
            message.success(`${e.dataTransfer.files[0].name} file uploaded successfully.`);
            localStorage.setItem("uploadedFiles", e.dataTransfer.files[0])
            setUpload(true)
        },
    };

    return (
        <React.Fragment>
            <PageHeader title="File Uploader" />
            <Dragger {...fileProps}>
                <p className="ant-upload-drag-icon">
                    <FileAddFilled />
                </p>
                <p className="ant-upload-text">Click or drag file to this area to upload</p>
                <p className="ant-upload-hint">detailed description here... allow file extension ?</p>
            </Dragger>
        </React.Fragment>
    )
}